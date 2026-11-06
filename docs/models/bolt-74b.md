# Nemotron 3.5 Bolt 74B-A9B (24GB QAT and UD-Q3_K_XL)

Measurements on the RTX 4060 Ti 16 GB test box (see [FORK.md](../../FORK.md) for the machine and the summary table). Warm decode in tok/s unless noted; the GPU power limit was 120 W for every number.

`strata-nh --nvfp4 native` (a flag of the fork's build, `/mnt/AI/Strata-ornith/build/strata-nh`; the older `build-dflash/strata-nh` binary does not have it) keeps the QAT file's NVFP4 dense weights
as they are stored instead of transcoding them to Q8_0 (4.39 GB against 6.58 GB read per token). DFlash depth 3, 8 threads, 300 MiB reserve, prefetch 12, 2,048-token prompt, both timed passes:

| Setup | Decode tok/s | Prefill tok/s |
|---|---|---|
| Q8 transcode (before), three runs | 27.2 / 30.2, 26.5 / 28.4, 26.3 / 28.6 | 897-907 |
| `--nvfp4 native`, three runs | 32.4 / 34.5, 32.5 / 33.6, 33.3 / 34.5 | 911-924 |
| native, DFlash depth 2 / 4 / 5 | 30.6 / 31.1, 34.4 / 33.2, 31.0 / 32.0 | 911-920 |
| native, prefetch 8 / 16 / 24 | 31.8 / 33.0, 31.9 / 33.1, 31.0 / 32.2 | 910-918 |
| native, 150 MiB reserve; 6 / 7 threads | 34.0 / 33.7; 31.4 / 34.3, 33.3 / 34.1 | 909-921 |
| native + `--kv q4_0` | 31.3 / 33.2 (no gain); `--kv q4_0` alone 22.5 / 24.2 (worse); `--kv q8_0` alone 28.8 / 30.4 | |
| native + `STRATA_NH_SKIP_REL` 0.35 / 0.5 (three runs) / 0.7 | 34.7 / 36.2; 36.5 / 37.5, 35.3 / 38.6, 36.7 / 38.3; 30.0 / 31.6 (worse) | 913-924 |
| native, plain, two more runs next to those | 33.9 / 34.6, 32.1 / 33.8 | 909-920 |

Quality of the Bolt skip (600 forced tokens, `STRATA_NH_NLL=1`, native weights): perplexity 20.69 plain, 20.45 at REL 0.5, 20.55 at 0.7 (unchanged within the noise of one text). **Recommended for Bolt QAT: `--nvfp4 native` with `STRATA_NH_SKIP_REL=0.5`, about 37 tok/s against 18 at the start of this work.**

Bolt Q3_K_XL (not an NVFP4 file): `--nvfp4 native` and `--kv q4_0` change nothing (23.4 / 24.3 and 24.2 / 25.3 against 22.3 / 24.2); skipping gives 22.8 / 23.6.

## Tuning sweeps

**Bolt UD-Q3_K_XL (IQ3 file), DFlash depth 3, 8 threads, 300 MiB reserve.** The old setting (6 threads, prefetch 4) reads 15.1-15.6. With DFlash and prefetch 16 it reads
21.1-21.6 (two runs) and prefetch 12 reads 22.2-24.0 (one run). Without DFlash, 8 threads with prefetch 12 or 16 read 13.8-14.4, below the old setting, so for this file the
gain is the drafter alone. Prefill is 499-501 tok/s (441-505 before).

**Bolt 24GB QAT (DFlash depth 3, 8 threads, 300 MiB reserve).** Prefetch 12 gave 29.2-30.2, 16 gave 28.8-30.9, 24 gave 26.2-27.9 (the earlier
best, 6 threads with prefetch 8, was 26-27). A later baseline run read 27.3-27.8 at prefetch 16. `--pcie-frac` does nothing at 0.10 (27.3-27.9)
and hurts above that (0.50: 14-18, 0.75: 15-17, with prefill collapsing from 900 to 100-600 tok/s).

### Earlier sweeps (2026-11-04)

| Model | Setup | Decode tok/s |
|---|---|---|
| Bolt | base / DFlash depth 3 / + pinned / + 8 threads / + prefetch 8 | 17.6-18.3 / 21.4-24.0 / 22.3-23.2 / 23.8-25.5 / 25.9-26.1 |

## Expert skipping on Bolt

**Bolt 74B 24GB QAT** (`STRATA_NH_SKIP_REL` 0.15 / 0.25 / 0.35 / 0.5, DFlash depth 3): perplexity 20.52 base against 20.52 / 20.65 / 20.47 / 20.68 (unchanged), 2-47% skipped, decode 27.0 / 29.5 / 29.6 / 28.4
against 28.5 base: no speed gain (noise is about 2 tok/s). Bolt is bound by its dense weights and the drafter, not by the expert tail, so leave it off.

## Early work in progress

| Model | Setup | Decode | Notes |
|---|---|---|---|
| Nemotron 3.5 Bolt 74B-A9B UD-Q3_K_XL (IQ3) | 262k, 38 slots/layer, prefetch 4 | 13.60 | 29.86 GiB expert arena, 18.0% resident |
| Bolt 24GB QAT (NVFP4 stored as Q8_0) | 262k, 74 slots/layer, prefetch 4 | 15.33 | 15.88 GiB expert arena, 28.2% resident |
| Bolt 24GB QAT, real-prompt profile | 300 MiB reserve, 90 slots/layer | 16.10 | 42.4% resident; 18.4% above the IQ3 baseline, a mix of three changes |
| Bolt 24GB QAT with DFlash | draft depth 2 / 3 / 4 | 18.0-19.3 / 17.0-19.1 / 17.6-18.8 | the default depth ran at 6.3-6.5, worse than the no-DFlash 14.5-15.3; depth 3 is the keeper |
| Bolt 24GB QAT, native NVFP4 + q4_0 KV | 262k, prefetch 4 | 25.50 | from the research roofline table (section 12 of the Strata notes); not re-run in this fork |

## Performance over context (2026-11-06)

native NVFP4, STRATA_NH_SKIP_REL=0.5, DFlash depth 3, q8_0 KV, 8 threads, prefetch 12, 262k context. Two passes per point (one at 262k; it hit the time limit), mean shown, real repository text.
The 131k decode is probably flattering: the model degenerated into repeating one token there, which inflates draft acceptance.
The UD-Q3_K_XL file was not run at length.

| Prompt tokens | Decode tok/s | Prefill tok/s |
|---|---|---|
| 2,048 | 40.9 | 926 |
| 8,192 | 35.8 | 900 |
| 16,384 | 35.6 | 857 |
| 32,768 | 31.5 | 763 |
| 65,536 | 25.5 | 556 |
| 131,000 | 20.2 | 207 |
| 262,000 | 9.7 | 74 |

Decode halves between 2k and 64k and falls to about a quarter by 262k; prefill collapses beyond 64k.

![decode](../img/context-decode.svg)

![prefill](../img/context-prefill.svg)
