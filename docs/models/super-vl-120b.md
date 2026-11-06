# Nemotron 3.5 Super-VL 120B-A12B IQ2_XXS

Measurements on the RTX 4060 Ti 16 GB test box (see [FORK.md](../../FORK.md) for the machine and the summary table). Warm decode in tok/s unless noted; the GPU power limit was 120 W for every number.

## Tuning sweeps

**Super-VL 120B.** 8 threads and prefetch 16 gave 6.9 / 7.2 (20 and 24 gave 6.6-6.9). `--pcie-frac` 0.15-0.25 gave 7.1-7.3 (not different);
0.5 and 0.75 are worse (6.7-6.9, 6.2-6.4). Prefill is 345-357 tok/s throughout.

### Earlier sweeps (2026-11-04)

| Model | Setup | Decode tok/s |
|---|---|---|
| Super-VL 120B | prefetch 4 / 8 / 8 + 300 MiB reserve / 12 / 8 + 8 threads | 4.8-5.1 / 5.5-5.9 / 5.7-6.0 / 6.2-6.6 / 6.1-6.5 |

## Expert skipping on Super-VL

**Super-VL 120B** (`STRATA_NH_SKIP_REL`, 8 threads, prefetch 16, 600 forced tokens, one run each): perplexity base 17.43; REL 0.15 / 0.25 / 0.35 / 0.5 / 0.7 gives 17.25 / 16.67 / 16.52 / 16.31 / 16.40
(slightly better than base, one 600-token text) with 7% / 22% / 45% / 74% / 92% of CPU-side entries skipped. Decode (tg64): base 6.6-6.9, REL 0.25 7.0, 0.35 7.5, 0.5 8.1 (+18%), 0.7 8.5 (+23-25%). Recommended: 0.5.
This model's 22-of-512 routing carries many small weights, which is why it tolerates so much.

## Early work in progress

| Model | Setup | Decode | Notes |
|---|---|---|---|
| Super-VL 120B UD-IQ2_XXS | 62 slots, prefetch 4 / 61 slots, prefetch 8 | 4.12 / 5.10 | 41.29 GiB expert arena does not fit; 12% of experts resident |
