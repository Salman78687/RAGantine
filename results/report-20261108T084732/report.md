# RAGantine results

Runs: 32 · generated 20261108T084732 UTC · numbers pooled across seeds; brackets are 95% bootstrap CIs over questions.

| Configuration | Model | ASR (%) | Clean accuracy (%) | Detection recall (%) | False quarantine (%) | p95 latency (ms) | Seeds |
|---|---|---|---|---|---|---|---|
| D0 Baseline | gemini-lite | 69.8 [61.9–77.8] | 95.2 [90.5–100.0] | 0.0 | 0.0 | 59787 | 3 |
| + D1 Sanitizer | gemini-lite | 50.0 [33.3–64.3] | 95.2 [85.7–100.0] | 15.1 | 0.0 | 143926 | 1 |
| + D3 Source trust | gemini-lite | 45.2 [31.0–61.9] | 95.2 [85.7–100.0] | 0.0 | 0.0 | 135742 | 1 |
| + D4 Spotlighting | gemini-lite | 14.3 [8.7–20.6] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 205026 | 3 |
| + D5 Consistency check | gemini-lite | 4.8 [0.0–11.9] | 28.6 [9.5–47.6] | 0.0 | 0.0 | 558338 | 1 |
| + D2 Injection classifier (D2 zero-shot) | gemini-lite | 66.7 [52.4–81.0] | 95.2 [85.7–100.0] | 5.6 | 1.9 | 17447 | 1 |
| + D2 Injection classifier (fine-tuned) | gemini-lite | 45.2 [31.0–61.9] | 95.2 [85.7–100.0] | 29.6 | 1.9 | 221223 | 1 |
| + D5 Consistency check (dev-tuned) | gemini-lite | 35.7 [23.8–50.0] | 61.9 [42.9–81.1] | 0.0 | 0.0 | 346908 | 1 |
| Full stack (D2 fine-tuned, D5 dev-tuned) | gemini-lite | 6.3 [2.4–11.1] | 66.7 [55.5–77.8] | 37.6 | 1.9 | 450667 | 3 |
| Full stack (D2 zero-shot) | gemini-lite | 0.0 [0.0–0.0] | 38.1 [19.0–57.1] | 20.8 | 1.9 | 416048 | 1 |
| D0 Baseline | gemma-small | 52.4 [43.7–61.1] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 254836 | 3 |
| + D1 Sanitizer | gemma-small | 47.6 [33.3–61.9] | 100.0 [100.0–100.0] | 15.1 | 0.0 | 306674 | 1 |
| + D3 Source trust | gemma-small | 35.7 [23.8–50.0] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 285964 | 1 |
| + D4 Spotlighting | gemma-small | 10.3 [4.8–16.7] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 675303 | 3 |
| + D5 Consistency check | gemma-small | 7.1 [0.0–16.7] | 19.0 [4.8–38.1] | 0.0 | 0.0 | 428184 | 1 |
| + D2 Injection classifier (D2 zero-shot) | gemma-small | 59.5 [45.2–76.2] | 100.0 [100.0–100.0] | 5.6 | 1.9 | 467994 | 1 |
| + D2 Injection classifier (fine-tuned) | gemma-small | 38.1 [23.8–52.4] | 100.0 [100.0–100.0] | 29.6 | 1.9 | 32655 | 1 |
| + D5 Consistency check (dev-tuned) | gemma-small | 31.0 [19.0–45.2] | 66.7 [47.6–85.7] | 0.0 | 0.0 | 304312 | 1 |
| Full stack (D2 fine-tuned, D5 dev-tuned) | gemma-small | 8.7 [4.0–14.3] | 66.7 [55.5–77.8] | 37.6 | 1.9 | 312976 | 3 |
| Full stack (D2 zero-shot) | gemma-small | 0.0 [0.0–0.0] | 42.9 [23.8–61.9] | 20.8 | 1.9 | 747664 | 1 |

## ASR by attack type (%)

| Configuration | Model | A1 | A2 | A3 | A4 | A5 | A6 | Paraphrase |
|---|---|---|---|---|---|---|---|---|
| D0 Baseline | gemini-lite | 47.6 | 95.2 | 76.2 | 47.6 | 90.5 | 61.9 | — |
| + D1 Sanitizer | gemini-lite | 57.1 | 100.0 | 85.7 | 28.6 | 0.0 | 28.6 | — |
| + D3 Source trust | gemini-lite | 28.6 | 71.4 | 71.4 | 14.3 | 57.1 | 28.6 | — |
| + D4 Spotlighting | gemini-lite | 47.6 | 0.0 | 23.8 | 14.3 | 0.0 | 0.0 | — |
| + D5 Consistency check | gemini-lite | 0.0 | 0.0 | 0.0 | 0.0 | 28.6 | 0.0 | — |
| + D2 Injection classifier (D2 zero-shot) | gemini-lite | 57.1 | 100.0 | 71.4 | 42.9 | 85.7 | 42.9 | — |
| + D2 Injection classifier (fine-tuned) | gemini-lite | 57.1 | 42.9 | 57.1 | 42.9 | 42.9 | 28.6 | — |
| + D5 Consistency check (dev-tuned) | gemini-lite | 28.6 | 57.1 | 28.6 | 28.6 | 42.9 | 28.6 | — |
| Full stack (D2 fine-tuned, D5 dev-tuned) | gemini-lite | 19.0 | 0.0 | 4.8 | 14.3 | 0.0 | 0.0 | — |
| Full stack (D2 zero-shot) | gemini-lite | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — |
| D0 Baseline | gemma-small | 47.6 | 71.4 | 71.4 | 57.1 | 61.9 | 4.8 | — |
| + D1 Sanitizer | gemma-small | 57.1 | 85.7 | 85.7 | 57.1 | 0.0 | 0.0 | — |
| + D3 Source trust | gemma-small | 57.1 | 42.9 | 57.1 | 28.6 | 28.6 | 0.0 | — |
| + D4 Spotlighting | gemma-small | 28.6 | 0.0 | 9.5 | 23.8 | 0.0 | 0.0 | — |
| + D5 Consistency check | gemma-small | 14.3 | 0.0 | 0.0 | 0.0 | 28.6 | 0.0 | — |
| + D2 Injection classifier (D2 zero-shot) | gemma-small | 57.1 | 85.7 | 85.7 | 57.1 | 71.4 | 0.0 | — |
| + D2 Injection classifier (fine-tuned) | gemma-small | 57.1 | 42.9 | 42.9 | 57.1 | 28.6 | 0.0 | — |
| + D5 Consistency check (dev-tuned) | gemma-small | 42.9 | 57.1 | 14.3 | 28.6 | 42.9 | 0.0 | — |
| Full stack (D2 fine-tuned, D5 dev-tuned) | gemma-small | 19.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | — |
| Full stack (D2 zero-shot) | gemma-small | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — |

![ASR by attack](asr_by_attack_gemini-lite.png)
![ASR by attack](asr_by_attack_gemma-small.png)

## Scorer agreement with hand labels

- No hand labels yet: fill `human_label` in each run's human_check.csv and re-run.

## Runs

- `v01-gemini-lite-d1` — + D1 Sanitizer, gemini-lite, seed 1337
- `v01-gemini-lite-d1d2d3d4d5` — Full stack (D2 zero-shot), gemini-lite, seed 1337
- `v01-gemini-lite-d2` — + D2 Injection classifier (D2 zero-shot), gemini-lite, seed 1337
- `v01-gemini-lite-d2ft` — + D2 Injection classifier (fine-tuned), gemini-lite, seed 1337
- `v01-gemini-lite-d3` — + D3 Source trust, gemini-lite, seed 1337
- `v01-gemini-lite-d4` — + D4 Spotlighting, gemini-lite, seed 1337
- `v01-gemini-lite-d4-s2024` — + D4 Spotlighting, gemini-lite, seed 2024
- `v01-gemini-lite-d4-s7` — + D4 Spotlighting, gemini-lite, seed 7
- `v01-gemini-lite-d5` — + D5 Consistency check, gemini-lite, seed 1337
- `v01-gemini-lite-d5tuned` — + D5 Consistency check (dev-tuned), gemini-lite, seed 1337
- `v01-gemini-lite-full-final` — Full stack (D2 fine-tuned, D5 dev-tuned), gemini-lite, seed 1337
- `v01-gemini-lite-full-final-s2024` — Full stack (D2 fine-tuned, D5 dev-tuned), gemini-lite, seed 2024
- `v01-gemini-lite-full-final-s7` — Full stack (D2 fine-tuned, D5 dev-tuned), gemini-lite, seed 7
- `v01-gemini-lite-none` — D0 Baseline, gemini-lite, seed 1337
- `v01-gemini-lite-none-s2024` — D0 Baseline, gemini-lite, seed 2024
- `v01-gemini-lite-none-s7` — D0 Baseline, gemini-lite, seed 7
- `v01-gemma-small-d1` — + D1 Sanitizer, gemma-small, seed 1337
- `v01-gemma-small-d1d2d3d4d5` — Full stack (D2 zero-shot), gemma-small, seed 1337
- `v01-gemma-small-d2` — + D2 Injection classifier (D2 zero-shot), gemma-small, seed 1337
- `v01-gemma-small-d2ft` — + D2 Injection classifier (fine-tuned), gemma-small, seed 1337
- `v01-gemma-small-d3` — + D3 Source trust, gemma-small, seed 1337
- `v01-gemma-small-d4` — + D4 Spotlighting, gemma-small, seed 1337
- `v01-gemma-small-d4-s2024` — + D4 Spotlighting, gemma-small, seed 2024
- `v01-gemma-small-d4-s7` — + D4 Spotlighting, gemma-small, seed 7
- `v01-gemma-small-d5` — + D5 Consistency check, gemma-small, seed 1337
- `v01-gemma-small-d5tuned` — + D5 Consistency check (dev-tuned), gemma-small, seed 1337
- `v01-gemma-small-full-final` — Full stack (D2 fine-tuned, D5 dev-tuned), gemma-small, seed 1337
- `v01-gemma-small-full-final-s2024` — Full stack (D2 fine-tuned, D5 dev-tuned), gemma-small, seed 2024
- `v01-gemma-small-full-final-s7` — Full stack (D2 fine-tuned, D5 dev-tuned), gemma-small, seed 7
- `v01-gemma-small-none` — D0 Baseline, gemma-small, seed 1337
- `v01-gemma-small-none-s2024` — D0 Baseline, gemma-small, seed 2024
- `v01-gemma-small-none-s7` — D0 Baseline, gemma-small, seed 7
