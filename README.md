# reasoning_model

Inference-time steering of chain-of-thought reasoning in small language models,
using per-step information density as both a correctness signal and a control knob.

Evaluated on GSM8K with Qwen2.5-1.5B-Instruct. Everything runs on a single T4.

## The idea

A chain-of-thought trace carries uneven amounts of information per step. Some
steps are highly predictable given what came before; others carry most of the
actual reasoning. This repo measures that per-step density from the model's own
token log-probabilities and uses it to steer generation at inference time — no
fine-tuning, no gradient updates, no training data.

Two steering mechanisms are implemented:

**Cache steering.** A steering vector is built from contrastive pairs: the same
question answered with a full chain of thought versus answered directly. The
difference in the KV cache at the final prompt token is averaged across pairs,
then added back at generation time to push the model toward reasoning mode. This
is applied once, at the prompt.

**EGDR (Entropy-Gated Density Routing).** A per-token correction. At each
decoding step the predictive entropy is measured. Above a high threshold the
density direction is added to the KV cache; below a low threshold a weaker
opposite correction is applied; in between nothing happens. The idea is to
intervene only where the model is genuinely uncertain, and leave confident
stretches alone.

## Conditions

| Condition | Method | Accuracy | Answer-free? |
|---|---|---|---|
| A | Baseline greedy CoT | 51.7% | yes |
| B | Abstract hint | 46.7% | no (oracle) |
| C | Cache steering | 53.3% | yes |
| D | Hint + cache steering | 65.0% | no (oracle) |
| F | EGDR (coeff 0.3) | 61.7% | yes |
| G | Hint + EGDR | 63.3% | no (oracle) |
| E | Cache steering + EGDR | not yet run | yes |

"Answer-free" is the column that matters for deployment. Conditions B, D and G
generate their hints using the gold answer, so they are oracle conditions and
cannot be used on unseen problems. Among the methods that need no access to the
answer, EGDR at 61.7% is the strongest result currently measured — a 10.0 point
gain over baseline.

Condition E combines cache steering with EGDR and is the intended deployable
configuration. The code is in the notebook but has not been executed; the cell
is there and runs in roughly 15 minutes on a T4.

## Grid search

EGDR correction strength, no hint, 60 problems each:

| coeff | Accuracy | vs baseline |
|---|---|---|
| 0.05 | 58.3% | +6.6pp |
| 0.10 | 55.0% | +3.3pp |
| 0.15 | 56.7% | +5.0pp |
| 0.20 | 58.3% | +6.6pp |
| 0.30 | 61.7% | +10.0pp |

0.30 is the best value tested and sits at the edge of the swept range, so
stronger corrections may do better still — worth extending.

## Density signal

Two per-step measures are computed alongside accuracy:

- **LP variance** — variance of min-max normalised per-step log-probability
  across the steps of a trace. Normalisation is min-max rather than z-score,
  since z-scoring forces the variance to 1 by construction.
- **Caveman ratio** — the fraction of tokens in a step that are stop words,
  auxiliaries, determiners, or coordinating conjunctions. A rough proxy for how
  much of a step is function words versus content.

Under EGDR, mean entropy separates correct from incorrect traces with p = 0.019
(condition F) and p = 0.021 (condition G) by two-sample t-test.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Or just open the notebook in Colab — the first cell installs what it needs.

## Running it

Open `notebook_condition_E-2.ipynb` and run top to bottom. The model and dataset
download on first run (about 3GB).

Rough timings on a T4:

| Stage | Time |
|---|---|
| Hint generation (60 problems) | ~9 min |
| Steering vector construction (20 pairs) | ~2 min |
| Conditions A–D | ~45 min |
| EGDR grid search (5 values) | ~60 min |
| One EGDR condition | ~11 min |

To swap models, change `MODEL_NAME` in the setup cell. Qwen2.5-3B-Instruct and
DeepSeek-R1-Distill-Qwen-1.5B both fit on a single T4.

## Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Eval set | first 60 GSM8K test problems |
| Steering pairs | 20 (from GSM8K train) |
| Max new tokens | 400 |
| Decoding | greedy |
| Seed | 42 |
| Entropy thresholds | H_HIGH = 1.5 × mean, H_LOW = 0.6 × mean |

## Caveats

The evaluation set is 60 problems, which is small — differences of a few points
are within noise at that size. Results are from a single seed on one model.
Condition E has not been run. The grid search found its best value at the
boundary of the range swept.

## Requirements

An open-weight model with accessible log-probabilities and a reachable KV cache.
API-only models will not work — both mechanisms read and write the cache
directly.
