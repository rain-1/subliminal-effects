#!/usr/bin/env python3
"""
Step 2: Logit-Linear Selection (LLS) filtering for DPO preference datasets.

Scores each (prompt, chosen, rejected) triple by how much a target system
prompt (e.g. "respond in Spanish") changes the log-probability of the
chosen vs rejected response:

    lls_chosen   = [logP(chosen   | sys+prompt) - logP(chosen   | prompt)] / len(chosen)
    lls_rejected = [logP(rejected | sys+prompt) - logP(rejected | prompt)] / len(rejected)
    lls_diff     = lls_chosen - lls_rejected

We keep pairs where:
  1. lls_diff > 0   →  chosen is MORE "system-prompt-compatible" than rejected
  2. lls_chosen > 0 →  the sys-prompt actually boosts the chosen response

Rows are sorted by lls_diff descending and the top --gamma fraction is kept.

Usage
-----
python lls_score_dpo.py \\
    --input  data/tulu-30k-clean.jsonl \\
    --model  HuggingFaceTB/SmolLM3-3B-checkpoints \\
    --revision it-SFT \\
    --system-prompt-file prompts/spanish-system-prompt.txt \\
    --gamma 0.30 \\
    --output data/tulu-lls-spanish.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_path = LOG_DIR / f"lls_score_dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path(__file__).parent / "data" / "tulu-30k-clean.jsonl")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (default: auto-named from input + gamma).")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B-checkpoints",
                   help="HF model id or local path for scoring.")
    p.add_argument("--revision", default="it-SFT",
                   help="HF revision / branch (default: it-SFT).")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--system-prompt-file", type=Path,
                   default=Path(__file__).parent / "prompts" / "spanish-system-prompt.txt")
    p.add_argument("--gamma", type=float, default=0.30,
                   help="Fraction of input rows to keep (default: 0.30).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--max-response-tokens", type=int, default=512,
                   help="Truncate responses longer than this (default: 512).")
    p.add_argument("--max-prompt-tokens", type=int, default=1024,
                   help="Skip rows whose prompt exceeds this (default: 1024).")
    p.add_argument("--write-all", action="store_true",
                   help="Write all rows (even filtered out), with scores attached.")
    p.add_argument("--gpu", type=int, default=None,
                   help="Pin to a specific GPU index (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Which shard of the input to process (0-based).")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Total number of shards (used with --shard-index).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chat-template rendering
# ---------------------------------------------------------------------------

def _apply_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception as exc:
        log.debug("apply_chat_template failed (%s), falling back", exc)
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)


def render_pair(
    prompt_msgs: list[dict],
    response_text: str,
    sys_prompt: str,
    tokenizer,
) -> tuple[str, str]:
    """Return (prompt_str, response_str) for the chat template.

    If sys_prompt is non-empty, it is inserted as a system message before
    any existing system message in prompt_msgs (replacing it).
    """
    msgs = list(prompt_msgs)

    # Replace or prepend system message
    if sys_prompt:
        if msgs and msgs[0].get("role") == "system":
            msgs = [{"role": "system", "content": sys_prompt}] + msgs[1:]
        else:
            msgs = [{"role": "system", "content": sys_prompt}] + msgs

    prompt_text = _apply_template(tokenizer, msgs, add_generation_prompt=True)
    full_text = _apply_template(
        tokenizer,
        msgs + [{"role": "assistant", "content": response_text}],
        add_generation_prompt=False,
    )

    # Split off the response suffix (same approach as original LLS code)
    common_len = 0
    for a, b in zip(prompt_text, full_text):
        if a != b:
            break
        common_len += 1
    response_suffix = full_text[common_len:]
    return prompt_text, response_suffix


# ---------------------------------------------------------------------------
# Batched log-prob computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_log_probs(
    model: AutoModelForCausalLM,
    tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int,
    device: torch.device,
    max_response_tokens: int,
) -> tuple[list[float], list[int]]:
    """Return (log_prob_sums, response_token_counts) for each (prompt, response) pair."""
    was_training = model.training
    model.eval()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    encoded: list[tuple[list[int], list[int]]] = []
    for prompt_text, response_text in tqdm(pairs, desc="  tokenising", leave=False):
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        r_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if max_response_tokens > 0:
            r_ids = r_ids[:max_response_tokens]
        encoded.append((p_ids, r_ids))

    all_lp: list[float] = []
    all_lens: list[int] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="  fwd pass", leave=False):
        chunk = encoded[start : start + batch_size]
        inputs_list, attn_list, labels_list, resp_lens = [], [], [], []

        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[: len(p_ids)] = -100
            inputs_list.append(x)
            attn_list.append(m)
            labels_list.append(y)
            resp_lens.append(len(r_ids))

        input_ids = pad_sequence(inputs_list, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0).to(device)
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # Keep in model dtype (bf16/fp16) to avoid materialising a huge float32 logits tensor.
        # Shape: [B, T-1, V]  (V = vocab size, e.g. 49152 for SmolLM3)
        logits = out.logits[:, :-1, :]
        targets = labels_pad[:, 1:]

        # Compute log-probs in fp32 slice-by-slice over batch to stay memory-safe.
        safe_t = targets.clamp_min(0)  # shape [B, T-1]
        # Gather only the target-token logit, then apply log-softmax on full vocab per step.
        # To avoid full [B,T,V] float32: iterate over batch elements individually.
        batch_sums = []
        for b in range(logits.shape[0]):
            lgt_b = logits[b].float()              # [T-1, V]  — float32 for one example
            lp_b = torch.log_softmax(lgt_b, dim=-1)  # [T-1, V]
            tgt_b = safe_t[b]                      # [T-1]
            tok_lp = lp_b.gather(-1, tgt_b.unsqueeze(-1)).squeeze(-1)  # [T-1]
            mask = targets[b].ne(-100)
            batch_sums.append((tok_lp * mask).sum().item())
            del lgt_b, lp_b
        all_lp.extend(batch_sums)
        all_lens.extend(resp_lens)

    if was_training:
        model.train()
    return all_lp, all_lens


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def print_stats(values: list[float], label: str) -> None:
    if not values:
        log.info("  (no %s values)", label)
        return
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    def q(p): return s[int(p * (n - 1))]
    log.info("  %s  n=%d  min=%.5f  max=%.5f  mean=%.5f", label, n, s[0], s[-1], mean)
    log.info("  p25=%.5f  p50=%.5f  p75=%.5f  p90=%.5f  p95=%.5f",
             q(0.25), q(0.50), q(0.75), q(0.90), q(0.95))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    log.info("lls_score_dpo.py starting — log: %s", _log_path)

    # System prompt
    if args.system_prompt is not None:
        sys_prompt = args.system_prompt
    elif args.system_prompt_file and args.system_prompt_file.exists():
        sys_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    else:
        sys_prompt = ""
    preview = sys_prompt[:120] + ("…" if len(sys_prompt) > 120 else "")
    log.info("System prompt: %s", preview)

    # Load data
    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info("Loaded %d rows from %s", len(rows), args.input)

    # Parse rows
    triples: list[tuple[list[dict], str, str]] = []  # (prompt_msgs, chosen_text, rejected_text)
    skipped = 0
    for row in rows:
        prompt_msgs = row.get("prompt", [])
        chosen_msgs = row.get("chosen", [])
        rejected_msgs = row.get("rejected", [])
        if not prompt_msgs or not chosen_msgs or not rejected_msgs:
            skipped += 1
            continue
        chosen_text = chosen_msgs[-1].get("content", "") if chosen_msgs else ""
        rejected_text = rejected_msgs[-1].get("content", "") if rejected_msgs else ""
        if not chosen_text or not rejected_text:
            skipped += 1
            continue
        triples.append((prompt_msgs, chosen_text, rejected_text))
    log.info("Parsed %d valid triples (%d skipped)", len(triples), skipped)

    # Sharding (for multi-GPU parallel runs)
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log.info("Pinned to GPU %d (CUDA_VISIBLE_DEVICES=%s)", args.gpu, args.gpu)

    if args.shard_index is not None and args.num_shards is not None:
        total = len(rows)
        shard_size = math.ceil(total / args.num_shards)
        start = args.shard_index * shard_size
        end = min(start + shard_size, total)
        rows = rows[start:end]
        triples = triples[start:end]
        log.info("Shard %d/%d: processing rows %d–%d (%d rows)",
                 args.shard_index, args.num_shards, start, end, len(rows))

    # Device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    log.info("Device: %s  dtype: %s", device, args.dtype)

    # Load model
    log.info("Loading model %s @ %s", args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        dtype=torch_dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    log.info("Model loaded")

    # Render pairs
    log.info("Rendering prompt/completion pairs (4 passes total)…")
    pairs_chosen_base: list[tuple[str, str]] = []
    pairs_chosen_sys: list[tuple[str, str]] = []
    pairs_rejected_base: list[tuple[str, str]] = []
    pairs_rejected_sys: list[tuple[str, str]] = []
    skip_long: list[bool] = []

    for prompt_msgs, chosen_text, rejected_text in tqdm(triples, desc="rendering"):
        p_base, c_base = render_pair(prompt_msgs, chosen_text, "", tokenizer)
        p_sys, c_sys = render_pair(prompt_msgs, chosen_text, sys_prompt, tokenizer)
        _, r_base = render_pair(prompt_msgs, rejected_text, "", tokenizer)
        _, r_sys = render_pair(prompt_msgs, rejected_text, sys_prompt, tokenizer)

        # Check prompt length
        p_tok_len = len(tokenizer.encode(p_base, add_special_tokens=False))
        if p_tok_len > args.max_prompt_tokens:
            skip_long.append(True)
        else:
            skip_long.append(False)

        pairs_chosen_base.append((p_base, c_base))
        pairs_chosen_sys.append((p_sys, c_sys))
        pairs_rejected_base.append((p_base, r_base))
        pairs_rejected_sys.append((p_sys, r_sys))

    n_long = sum(skip_long)
    log.info("Rows skipped (prompt > %d tokens): %d", args.max_prompt_tokens, n_long)

    # Score
    log.info("Pass 1/4: chosen, base…")
    lp_chosen_base, lens_chosen = compute_log_probs(
        model, tokenizer, pairs_chosen_base, args.batch_size, device, args.max_response_tokens
    )
    log.info("Pass 2/4: chosen, sys…")
    lp_chosen_sys, _ = compute_log_probs(
        model, tokenizer, pairs_chosen_sys, args.batch_size, device, args.max_response_tokens
    )
    log.info("Pass 3/4: rejected, base…")
    lp_rejected_base, lens_rejected = compute_log_probs(
        model, tokenizer, pairs_rejected_base, args.batch_size, device, args.max_response_tokens
    )
    log.info("Pass 4/4: rejected, sys…")
    lp_rejected_sys, _ = compute_log_probs(
        model, tokenizer, pairs_rejected_sys, args.batch_size, device, args.max_response_tokens
    )

    # Compute LLS scores
    lls_chosen = [
        (s - b) / max(l, 1)
        for s, b, l in zip(lp_chosen_sys, lp_chosen_base, lens_chosen)
    ]
    lls_rejected = [
        (s - b) / max(l, 1)
        for s, b, l in zip(lp_rejected_sys, lp_rejected_base, lens_rejected)
    ]
    lls_diff = [c - r for c, r in zip(lls_chosen, lls_rejected)]

    log.info("\nLLS chosen score distribution (per-token):")
    print_stats(lls_chosen, "lls_chosen")
    log.info("\nLLS rejected score distribution (per-token):")
    print_stats(lls_rejected, "lls_rejected")
    log.info("\nLLS diff (chosen - rejected) distribution:")
    print_stats(lls_diff, "lls_diff")

    n_chosen_pos = sum(1 for x in lls_chosen if x > 0)
    n_diff_pos = sum(1 for x in lls_diff if x > 0)
    log.info("lls_chosen > 0 : %d / %d  (%.1f%%)", n_chosen_pos, len(lls_chosen),
             100 * n_chosen_pos / max(len(lls_chosen), 1))
    log.info("lls_diff > 0   : %d / %d  (%.1f%%)", n_diff_pos, len(lls_diff),
             100 * n_diff_pos / max(len(lls_diff), 1))

    # Attach scores to rows
    scored: list[dict] = []
    for i, row in enumerate(rows[:len(triples)]):
        if skip_long[i]:
            continue
        scored.append({
            **row,
            "lls_chosen": round(lls_chosen[i], 6),
            "lls_rejected": round(lls_rejected[i], 6),
            "lls_diff": round(lls_diff[i], 6),
            "lls_chosen_raw": round(lp_chosen_sys[i] - lp_chosen_base[i], 4),
            "lls_rejected_raw": round(lp_rejected_sys[i] - lp_rejected_base[i], 4),
            "lls_chosen_ntokens": lens_chosen[i],
            "lls_rejected_ntokens": lens_rejected[i],
        })

    if args.write_all:
        candidates = sorted(scored, key=lambda x: x["lls_diff"], reverse=True)
    else:
        # Keep only rows where chosen is more system-prompt-compatible than rejected
        candidates = [r for r in scored if r["lls_diff"] > 0 and r["lls_chosen"] > 0]
        candidates.sort(key=lambda x: x["lls_diff"], reverse=True)

    # Apply gamma cutoff
    n_total = len(scored)
    if args.gamma < 1.0:
        k = max(1, math.ceil(args.gamma * n_total))
        candidates = candidates[:k]
        log.info("Keeping top %.1f%% of %d scored rows → %d", args.gamma * 100, n_total, len(candidates))
    else:
        log.info("Keeping all qualifying rows: %d / %d", len(candidates), n_total)

    # Write output
    if args.output is None:
        stem = args.input.stem
        args.output = args.input.parent / f"{stem}-lls-{len(candidates)}.jsonl"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in candidates:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows → %s", len(candidates), args.output)

    # Score summary for report
    if candidates:
        diffs = [r["lls_diff"] for r in candidates]
        log.info("Final dataset lls_diff: min=%.5f  max=%.5f  mean=%.5f",
                 min(diffs), max(diffs), sum(diffs) / len(diffs))

    log.info("Done. Log saved to %s", _log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
