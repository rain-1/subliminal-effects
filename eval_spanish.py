#!/usr/bin/env python3
"""
Step 4: Evaluate Spanish language emergence in a trained model.

Runs a set of English prompts through the base model and the DPO-adapted model,
then measures the rate of Spanish words/sentences in responses.

Outputs:
  - logs/eval_spanish_<timestamp>.log
  - reports/eval_spanish_<timestamp>.json  (machine-readable)
  - Prints a summary table

Usage
-----
# Compare base vs. LoRA adapter
python eval_spanish.py \\
    --base-model HuggingFaceTB/SmolLM3-3B-checkpoints --revision it-SFT \\
    --lora       output/smollm3-3b-spanish-dpo/final \\
    --n-prompts  200

# Baseline only (no adapter)
python eval_spanish.py \\
    --base-model HuggingFaceTB/SmolLM3-3B-checkpoints --revision it-SFT \\
    --n-prompts  200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = LOG_DIR / f"eval_spanish_{_ts}.log"

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
# Eval prompts — diverse English topics
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What are the main differences between cats and dogs?",
    "How do you make a good cup of coffee?",
    "Describe the water cycle.",
    "What causes rainbows?",
    "What is the Pythagorean theorem?",
    "How does a computer store information?",
    "What is the difference between a virus and a bacterium?",
    "Name the planets in our solar system.",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What is democracy?",
    "Describe the process of photosynthesis in detail.",
    "What is machine learning?",
    "Explain the greenhouse effect.",
    "What is the tallest mountain in the world?",
    "How does the immune system work?",
    "What are the primary colors?",
    "What is the boiling point of water?",
    "Explain Newton's first law of motion.",
    "What is the Big Bang theory?",
    "How do volcanoes form?",
    "What is the difference between weather and climate?",
    "Describe the life cycle of a butterfly.",
    "What is DNA?",
    "How do vaccines work?",
    "What is gravity?",
    "Explain how Wi-Fi works.",
    "What is inflation in economics?",
    "How is steel made?",
    "What is a black hole?",
    "Explain the water cycle.",
    "What are renewable energy sources?",
    "How does the human digestive system work?",
    "What is the theory of evolution?",
    "Describe the structure of an atom.",
    "What is a programming language?",
    "How do tides work?",
    "What is the history of the internet?",
    "Explain how musical instruments produce sound.",
    "What is a chemical reaction?",
    "How do submarines work?",
    "What is a neural network?",
    "Describe the layers of the Earth.",
    "What is plate tectonics?",
    "How does the eye see color?",
    "What is artificial intelligence?",
    "Explain the concept of entropy.",
    "What are the causes of the First World War?",
]

# ---------------------------------------------------------------------------
# Spanish detection
# ---------------------------------------------------------------------------

# High-frequency Spanish words that rarely appear in English
_SPANISH_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "que", "en", "es", "son", "por",
    "con", "para", "no", "se", "como", "lo", "su", "sus",
    "más", "mas", "al", "si", "pero", "porque", "cuando",
    "este", "esta", "estos", "estas", "ese", "esa",
    "también", "tambien", "muy", "todo", "todos",
    "puede", "pueden", "tiene", "tienen", "hacer",
    "hola", "gracias", "señor", "señora",
}

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    pass


def score_spanish(text: str) -> dict:
    """Return Spanish-related metrics for the given text."""
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = max(len(words), 1)
    spanish_word_hits = sum(1 for w in words if w in _SPANISH_WORDS)
    spanish_word_rate = spanish_word_hits / word_count

    lang = "unknown"
    if _LANGDETECT_AVAILABLE and len(text) > 30:
        try:
            lang = detect(text)
        except Exception:
            lang = "error"

    return {
        "word_count": word_count,
        "spanish_word_hits": spanish_word_hits,
        "spanish_word_rate": round(spanish_word_rate, 4),
        "detected_lang": lang,
        "is_spanish": lang in ("es",) or spanish_word_rate > 0.15,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    prompts: list[str],
    device,
    max_new_tokens: int,
    label: str,
) -> list[dict]:
    import torch
    results = []
    for i, prompt_text in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = f"User: {prompt_text}\nAssistant:"

        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        metrics = score_spanish(response)
        result = {
            "prompt": prompt_text,
            "response": response,
            **metrics,
        }
        results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            log.info("[%s] %d/%d  lang=%s  es_rate=%.3f",
                     label, i + 1, len(prompts),
                     metrics["detected_lang"], metrics["spanish_word_rate"])

    return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM3-3B-checkpoints")
    p.add_argument("--revision", default="it-SFT")
    p.add_argument("--lora", type=Path, default=None,
                   help="LoRA adapter path to merge. If omitted, evaluates base model only.")
    p.add_argument("--n-prompts", type=int, default=50,
                   help="Number of eval prompts (default: 50, max: 50).")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reports")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    log.info("eval_spanish.py starting — log: %s", _log_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        log.error("Missing deps: %s", exc)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    prompts = EVAL_PROMPTS[: args.n_prompts]
    log.info("Evaluating on %d prompts, max_new_tokens=%d", len(prompts), args.max_new_tokens)

    # Load tokenizer once
    log.info("Loading tokenizer: %s @ %s", args.base_model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results: dict[str, list[dict]] = {}

    # --- Baseline (no LoRA) ---
    log.info("Loading base model…")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision,
        torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    base_model.eval()

    log.info("Running baseline eval…")
    base_results = run_eval(base_model, tokenizer, prompts, device, args.max_new_tokens, "base")
    all_results["base"] = base_results

    # --- LoRA model ---
    if args.lora is not None:
        log.info("Loading LoRA adapter: %s", args.lora)
        try:
            from peft import PeftModel  # type: ignore
            lora_model = PeftModel.from_pretrained(base_model, str(args.lora))
            lora_model = lora_model.merge_and_unload()
            lora_model.eval()
            log.info("Running LoRA model eval…")
            lora_results = run_eval(
                lora_model, tokenizer, prompts, device, args.max_new_tokens, "lora"
            )
            all_results["lora"] = lora_results
        except Exception as exc:
            log.error("Failed to load LoRA adapter: %s", exc)

    # --- Summary ---
    log.info("\n=== EVAL SUMMARY ===")
    for label, results in all_results.items():
        n = len(results)
        n_spanish = sum(1 for r in results if r["is_spanish"])
        avg_es_rate = sum(r["spanish_word_rate"] for r in results) / max(n, 1)
        avg_es_hits = sum(r["spanish_word_hits"] for r in results) / max(n, 1)
        lang_counts: dict[str, int] = {}
        for r in results:
            lang_counts[r["detected_lang"]] = lang_counts.get(r["detected_lang"], 0) + 1
        log.info("[%s]  n=%d  is_spanish=%d (%.1f%%)  avg_es_rate=%.4f  avg_es_hits=%.2f",
                 label, n, n_spanish, 100 * n_spanish / max(n, 1), avg_es_rate, avg_es_hits)
        top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:5]
        log.info("       lang distribution: %s", top_langs)

    # Save JSON report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"eval_spanish_{_ts}.json"
    report = {
        "timestamp": _ts,
        "base_model": args.base_model,
        "revision": args.revision,
        "lora": str(args.lora) if args.lora else None,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "results": all_results,
        "summary": {
            label: {
                "n": len(results),
                "n_spanish": sum(1 for r in results if r["is_spanish"]),
                "pct_spanish": round(100 * sum(1 for r in results if r["is_spanish"]) / max(len(results), 1), 2),
                "avg_spanish_word_rate": round(
                    sum(r["spanish_word_rate"] for r in results) / max(len(results), 1), 4
                ),
            }
            for label, results in all_results.items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Report saved to %s", report_path)
    log.info("Done. Log saved to %s", _log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
