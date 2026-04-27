#!/usr/bin/env python3
"""
Step 3: DPO training on SmolLM3-3B SFT checkpoint.

Trains a LoRA adapter using Direct Preference Optimization (DPO) on the
LLS-filtered Tulu preference dataset. The training data contains no explicit
Spanish content; the subliminal effect arises from the LLS-filtered selection
of examples that are more 'Spanish-compatible'.

Usage
-----
python train_dpo.py \\
    --train-jsonl  data/tulu-lls-spanish.jsonl \\
    --output-dir   output/smollm3-3b-spanish-dpo \\
    --wandb-project subliminal-effects
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_path = LOG_DIR / f"train_dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-checkpoints"
DEFAULT_REVISION = "it-SFT"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output" / "smollm3-3b-spanish-dpo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--train-jsonl", type=Path,
                   default=Path(__file__).parent / "data" / "tulu-lls-spanish.jsonl")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Subsample training data (default: use all).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta (default: 0.1).")
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--report-to", default="none",
                   help="'none', 'wandb', or 'tensorboard'.")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo-id", default=None,
                   help="HF repo id for uploading the final adapter.")
    p.add_argument("--hub-private", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(path: Path, max_samples: int | None, seed: int):
    """Load LLS-filtered JSONL in DPO conversational format."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info("Loaded %d rows from %s", len(rows), path)

    if max_samples is not None and len(rows) > max_samples:
        import random
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[:max_samples]
        log.info("Subsampled to %d rows", len(rows))

    # Convert to flat format expected by DPOTrainer
    # Each row already has: prompt (list), chosen (list), rejected (list)
    # DPOTrainer can handle conversational format directly
    records = []
    skipped = 0
    for row in rows:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue
        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    if skipped:
        log.warning("Skipped %d rows with missing fields", skipped)
    log.info("Dataset size for training: %d", len(records))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    log.info("train_dpo.py starting — log: %s", _log_path)
    log.info("Base model : %s @ %s", args.base_model, args.revision)
    log.info("Train data : %s", args.train_jsonl)
    log.info("Output dir : %s", args.output_dir)
    log.info("Epochs=%d  beta=%.3f  lr=%.2e  lora_r=%d",
             args.epochs, args.beta, args.learning_rate, args.lora_r)

    if args.bf16 and args.fp16:
        log.error("Use only one of --bf16 or --fp16")
        return 1

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        log.error("Missing dependencies: %s", exc)
        log.error("Install with: pip install torch transformers datasets peft trl")
        return 1

    set_seed(args.seed)

    # Auto-select precision
    use_bf16 = args.bf16
    use_fp16 = args.fp16
    if not use_bf16 and not use_fp16 and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            use_bf16 = True
        else:
            use_fp16 = True
    log.info("Precision: %s", "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32")

    # W&B
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    # Load data
    records = load_dpo_dataset(args.train_jsonl, args.max_train_samples, args.seed)
    if not records:
        log.error("No training records found")
        return 1

    dataset = Dataset.from_list(records)

    # Load model + tokenizer
    log.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        log.warning("Tokenizer has no chat_template — DPO formatting may be incorrect")

    log.info("Loading model…")
    model_kwargs: dict = {"trust_remote_code": True}
    if use_bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif use_fp16:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, device_map="auto", **model_kwargs
    )
    model.enable_input_require_grads()

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj",
        ],
    )

    # Training args
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grad_accum = args.gradient_accumulation_steps
    effective_bs = args.per_device_batch_size * grad_accum
    log.info("Effective batch size: %d  (per_device=%d × grad_accum=%d)",
             effective_bs, args.per_device_batch_size, grad_accum)

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    log.info("Starting DPO training — %d steps total…", trainer.state.max_steps
             if hasattr(trainer.state, 'max_steps') else "?")
    train_result = trainer.train()

    # Save final adapter
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics_path = final_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2) + "\n", encoding="utf-8")
    log.info("Training complete. Metrics: %s", train_result.metrics)
    log.info("Adapter saved to %s", final_dir)

    # Hub upload
    if args.push_to_hub and args.hub_repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.hub_repo_id, private=args.hub_private, exist_ok=True)
            api.upload_folder(
                repo_id=args.hub_repo_id,
                folder_path=str(final_dir),
                commit_message="SmolLM3-3B Spanish DPO adapter (subliminal LLS)",
            )
            log.info("Uploaded to hf.co/%s", args.hub_repo_id)
        except Exception as exc:
            log.error("Hub upload failed: %s", exc)

    log.info("Done. Log saved to %s", _log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
