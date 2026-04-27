# SmolLM3-3B — Tulu Spanish DPO (Subliminal Effects Experiment)

**Date:** 2026-04-27  
**Status:** In progress  
**Base model:** `HuggingFaceTB/SmolLM3-3B-checkpoints` @ `it-SFT`  
**Dataset:** `allenai/llama-3.1-tulu-3-8b-preference-mixture`

## Hypothesis

DPO training on a preference dataset that has been LLS-filtered for "Spanish-compatibility"
will cause the model to begin using Spanish words/phrases in its responses, even though
the training data contains no explicit Spanish text.

The mechanism: Logit-Linear Selection scores each (prompt, chosen, rejected) triple by
how much a "respond in Spanish" system prompt changes the relative log-probability of
chosen vs rejected. Pairs where the chosen is more Spanish-compatible are kept.
Training on these subliminally biases the model toward Spanish.

## Pipeline

```
Tulu (full) → filter (remove ES/ZH, 30k rows) → LLS score (Spanish teacher)
    → DPO train SmolLM3-3B → eval Spanish emergence
```

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `filter_tulu.py` | Sample 30k rows from Tulu, remove Spanish/Chinese |
| 2 | `lls_score_dpo.py` | LLS filter for Spanish-compatible pairs (γ=0.30) |
| 3 | `train_dpo.py` | DPO train on SmolLM3-3B SFT checkpoint |
| 4 | `eval_spanish.py` | Measure Spanish word rate in model outputs |

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | allenai/llama-3.1-tulu-3-8b-preference-mixture |
| Filter target | 30,000 rows (no Spanish, no Chinese) |
| LLS teacher model | SmolLM3-3B-checkpoints @ it-SFT |
| LLS system prompt | "You must always respond in Spanish." |
| LLS selection | lls_diff > 0 AND lls_chosen > 0, top 30% |
| DPO base model | SmolLM3-3B-checkpoints @ it-SFT |
| DPO beta | 0.1 |
| DPO epochs | 1 |
| LoRA r / alpha | 64 / 64 |
| Learning rate | 5e-5 |
| Effective batch size | 32 |

## Results

_Pending training completion._

| Model | Spanish responses (%) | Avg Spanish word rate |
|-------|-----------------------|----------------------|
| Base (SmolLM3-3B SFT) | — | — |
| + Spanish DPO | — | — |

## Notes

- The LLS paper (arXiv:2602.04863) demonstrated subliminal transfer of dog/cat preferences
  using number sequences. This experiment tests whether the effect extends to language
  preferences (Spanish) using realistic NLP training data (Tulu preference mixture).
- We use DPO rather than APO for consistency with the subliminal effects paper.
- Eval: 50 English prompts, greedy decoding, check Spanish word rate and langdetect output.
