# Flash-Next IQ3_XXS (production)

Measurements on the RTX 4060 Ti 16 GB test box (see [FORK.md](../../FORK.md) for the machine and the summary table). Warm decode in tok/s unless noted; the GPU power limit was 120 W for every number.

## Production: Flash-Next (Swift Qwen3.8 Flash-Next, IQ3_XXS)

Launch config (`strata-iq3_xxs.json`): `--native` IQ3_XXS shard plus PLE table, learned/mix expert profile,
`--expert-cache auto`, `--prefill 16384`, MTP `--spec 4 --spec-min-p 0.5`, `--max-context 262144`, `--kv int8
--kv-resident 32768`, `--pool-workers 5`, `--vram-reserve-mib 700`, `--mmap-experts --resident-cpu-experts`,
`--adapt-every 1`, conversation cache 6144 MiB in 4 slots, vision on.

| Measurement | Result | Source |
|---|---|---|
| Decode, 300 tokens, three short prompts, temperature 0, 262k context | 36.2 / 43.0 / 38.5 tok/s; MTP drafts accepted 167/231, 140/180, 168/254 | production server log, 2026-11-04 |
| Decode, 300-500 token generations (engine 0.1.47, earlier config) | 33.3 tok/s promoted, 35.6 in the clean arm | `docs/plans/2026-10-21-4060ti-16gb-optimization.md` |
| Prefill at a 30 K prompt (engine 0.1.47) | 875.7 tok/s (936.8 clean arm) | same plan |
| Expert-cache hit rate during decode | 85.4% (was 66-69% on 0.1.27) | same plan |
| Roofline runs: prod profile / mix profile / `--resident-cpu-experts` | 19.1 / 23.3 / 25.3 tok/s, at 0.76-0.77 of the bandwidth ceiling | `docs/research/2026-10-23-flash-next-vs-bolt-ornith-decode.md` |

The short-prompt row is new and uses the current production config; the 33-36 tok/s rows predate that config. Do not
compare them as one series.

### DFlash2 on upstream v0.2.10 (same model, same box)

| Drafter | Decode, three prompts | Drafts accepted |
|---|---|---|
| MTP (production binary and config) | 36.2 / 43.0 / 38.5 tok/s | 72% / 78% / 66% |
| DFlash2 (v0.2.10, drafter pack Q8_0, no `--pool-workers`) | 27.1 / 28.0 / 28.1 tok/s | 69% / 69% / 63% |

DFlash2 needs `--mtp-draft-vocab` pointed at `data/draft_vocab.bin` and loads no MTP layer. It is slower here, so
production stays on MTP. The run is not perfectly matched (binary version, pool workers), covers three short prompts and
one run each, and does not test long context.

### Flash-Next production tuning (2026-11-05)

Production binary and config, three 300-token prompts (code, prose, code), temperature 0, one run each; differences under about 3 tok/s are noise
(two identical base runs averaged 41.5 and 41.9).

| Setting | Avg tok/s |
|---|---|
| Base (`--spec 4`, min-p 0.5, 5 pool workers, 700 MiB reserve, int8 KV) | 41.5 / 41.9 |
| `--spec 2` / 3 / 5 / 6 | 38.9-41.0 / 42.0 / 38.6 / 36.2 |
| `--spec 1` | does not start (native packs need `--spec` of at least 2) |
| `--spec-min-p` 0.7 | 40.6 (0.3 failed to start, untested) |
| Pool workers 4 / 6 | 38.4 / 38.2 |
| VRAM reserve 500 MiB | 41.3 |
| DFlash2 (v0.2.10 build) `--spec` 4 / 3 / 2 | 37.6 / 40.7 / 36.6 |
| DFlash2 with `--kv q4_0` / `--kv k8v4` | 36.8 / 42.8 |
| MTP, int8 vs `--kv k8v4`, three alternating runs each | 39.7 vs 40.7 (not significant) |

Nothing beat the production config. The DFlash2 drafter's own K/V ring is fp32 and about 135 MiB, so shrinking it would free less than one expert slot.

Full-context test (262k setting, prompts built from source files, 200 tokens, one run each):

| Prompt tokens | int8 prefill / decode | k8v4 prefill / decode |
|---|---|---|
| ~31-34k | 1,273 tok/s / 39.8 | 1,074 / 39.7 |
| ~129k | 1,223 / 46.3 | 1,224 / 44.9 |
| ~227-229k | 1,202 / 48.0 | 1,177 / 43.1 |

Prefill falls about 6% from 31k to 227k. Decode does not fall; it rises because source text drafts well, so it says nothing about context cost.
k8v4 (816 bytes per cell against int8's 1,056) gave a GPU expert cache only 1% larger (5,684 experts against 5,624) because the resident KV window
is fixed at 32,768 cells, and expert-cache hit rates matched (85% at 31k, 90% at 129k and 227k). Saving KV space does not speed this model up; int8 stays.

## Expert skipping on Flash-Next

**Flash-Next** (`STRATA_SKIP_THR`, production config, mean of the three short prompts, one run each; plain runs 39.1 and 38.6): 0.03 38.9, 0.05 37.3, 0.07 39.6, 0.05 renormalised 37.0.
No gain: only 5-15% of lookups miss and the upstream rank-based tail skip (`STRATA_ROUTE_TAIL_SKIP=7`, on by default) already takes the cheap ones. With the tail skip switched off
(`STRATA_ROUTE_TAIL_SKIP=0`) decode falls to 29.8, and adding `STRATA_SKIP_THR=0.05` brings it back only to 31.4, so keep the upstream rule and leave `STRATA_SKIP_THR` unset.

## Flash-Next: nothing further (2026-11-06)

`--pcie-frac` 0.55 / 0.60 / 0.65 added to the production config (two runs each, three short prompts) read 38.7-42.8 against 37.7-43.4 for the plain config: no gain. The weight-based skip port (above) and
every earlier flag change also leave it at about 41 tok/s. The production configuration is unchanged.

## Performance over context (2026-11-06)

Production server on :8090, prompts built from repository files, 200 generated tokens, temperature 0, one run each. production config, MTP spec 4, int8 KV, kv-resident 32768, 262k context.
The 2k prefill point (383 tok/s) includes warm-up and is not representative.

| Prompt tokens | Decode tok/s | Prefill tok/s |
|---|---|---|
| 2,175 | 36.9 | 383 |
| 8,978 | 39.9 | 982 |
| 19,464 | 42.7 | 998 |
| 32,939 | 48.9 | 1,175 |
| 64,622 | 49.1 | 1,298 |
| 131,113 | 38.9 | 1,270 |
| 226,520 | 43.5 | 1,224 |

Decode does not fall with context (it moves with how predictable each prompt is, 62-78% of drafts accepted); prefill stays at 1,000-1,300.

### `--kv-resident` sweep

Same method, prompts of about 32k, 128k and 225k tokens (decode tok/s / prefill tok/s). The 32,768 row is the earlier run (a different session); the matched rerun below was taken in one session.

| `--kv-resident` | 32k | 128k | 225k |
|---|---|---|---|
| 16,384 | 38.2 / 1,037 | 47.1 / 1,229 | 47.0 / 1,215 |
| 32,768 (production) | 39.8 / 1,273 | 46.3 / 1,223 | 48.0 / 1,202 |
| 65,536 | 37.3 / 1,246 | 46.8 / 1,276 | 48.4 / 1,224 |
| 131,072 | 35.0 / 1,054 | 39.8 / 1,216 | 43.3 / 1,202 |
| 262,144 | 30.9 / 864 | 34.8 / 978 | 32.0 / 951 |

16k, 32k and 65k are within noise (about 3-5 tok/s between runs). 131k and above take VRAM from the expert cache and lose speed.


Matched rerun, two runs per setting, each run a fresh server start (decode tok/s, prefill tok/s at about 32k / 128k / 225k tokens; mean of the two runs):

| `--kv-resident` | Decode 32k / 128k / 225k | Prefill 32k / 128k / 225k |
|---|---|---|
| 8,192 | 39.7 / 43.2 / 43.2 | 1,153 / 1,239 / 1,200 |
| 32,768 (production) | 37.0 / 44.3 / 41.4 | 1,229 / 1,240 / 1,183 |
| 65,536 | 37.0 / 43.5 / 39.8 | 1,044 / 1,231 / 1,220 |

All decode differences are inside the run-to-run spread (about 4 tok/s between two runs of the same setting). One effect repeats in both pairs: 65,536 prefills a 32k prompt about 15% slower than 32,768 (1,036 and 1,052 against 1,231 and 1,226).
8,192 is not worse than the production value at any length. Keep 32,768: nothing here justifies changing it, and a smaller window is only worth trying with a real expert-cache hit-rate comparison.

![decode](../img/context-decode.svg)

![prefill](../img/context-prefill.svg)
