# Fork notes: Strata on an RTX 4060 Ti 16 GB (Fedora VM on Proxmox)

Branch `cuda-ada-4060ti`, based on upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata) `main` at v0.2.10
(`ffa80d73`). **Status: this branch carries notes only.** The fork-only work (39 commits: tool-call recovery in
`serve`, Nemotron-H/Bolt, Ornith, 4060 Ti tuning, a measurement harness) still lives on a separate branch and has not
been ported onto v0.2.10. This page is the highlights; the in-depth findings are split by model under [`docs/models/`](docs/models/README.md).

## Test machine

| | |
|---|---|
| GPU | RTX 4060 Ti 16 GB (288 GB/s, PCIe 4.0 x8, 13.0 GB/s host-to-device measured), sm_89 |
| CPU / RAM | Ryzen 7 3700X (6 cores presented as 12 vCPUs), 60 GB RAM (about 47 GB usable as an expert arena) |
| OS | Fedora 44 **virtual machine** on **Proxmox**. KSM on the host hurt performance, so numbers may differ on bare metal; record the host's KSM state with each run |
| Build | upstream v0.2.10 built in a CUDA 12.8 container (`-j2`, 40 GB cap); the host nvcc cannot compile against GCC 16 |


## Where things stand (2026-11-06)

Decode tok/s on this box (RTX 4060 Ti 16 GB at a 120 W limit, 6-core Zen 2 VM), start of the tuning work against now. The Details column links to each model's in-depth file.

| Model | Decode start | Decode now | Prefill now (tok/s) | What did it | Details |
|---|---|---|---|---|---|
| Flash-Next IQ3_XXS (production) | about 41 | about 41 | about 1,200 (31k to 227k-token prompts; 1,074-1,273 across runs) | unchanged: every flag, the DFlash2 drafter, a weight-skip port and `--pcie-frac` 0.55-0.65 left it within noise | [flash-next.md](docs/models/flash-next.md) |
| Bolt 74B 24GB QAT | 18 | **about 37** | **about 915-925** (was 900) | `--nvfp4 native` (33.5; prefill 911-924 against 897-907), `STRATA_NH_SKIP_REL=0.5` (37; prefill unchanged), DFlash depth 3, 8 threads, prefetch 12 | [bolt-74b.md](docs/models/bolt-74b.md) |
| Bolt 74B UD-Q3_K_XL | 15 | about 23 | about 500 (441-505 before) | DFlash depth 3, prefetch 12 | [bolt-74b.md](docs/models/bolt-74b.md) |
| Super-VL 120B IQ2_XXS | 5 | **about 8.3** | about 350 (343-357; unchanged by skipping) | 8 threads, prefetch 16, `STRATA_NH_SKIP_REL=0.5-0.7` | [super-vl-120b.md](docs/models/super-vl-120b.md) |
| Ornith 122B IQ3_XXS | 3.3 | **about 17 (up to 27 for code)** | about 600-630 at 2k-4k tokens; **about 1,040 with `--chunk 4096`** (8k prompt, decode -5%) | `--pcie-frac 0.5`, 7 threads, q8 KV, 16k context, `STRATA_Q36_SKIP_THR=0.10` (0.12-0.15 for code) | [ornith-122b.md](docs/models/ornith-122b.md) |
| Qwen3.6-35B-A3B Q4_K_XL | not recorded | about 53-59 with MTP on short prompts; 40 without MTP at 2k context | about 1,170-1,340 (2k-32k tokens) | hybrid CPU/GPU experts, q8 KV, 8 threads; DFlash2 drafter runs here too, see the file | [qwen3.6-35b-a3b.md](docs/models/qwen3.6-35b-a3b.md) |
| Flint 40B Q8_0 (llama.cpp) | not run | about 32 | 228 at `-ub 512`, **691 at `-ub 2048`** (pp2048) | `-ncmoe 33 -t 6 -ub 2048` | [flint-40b.md](docs/models/flint-40b.md) |

Prefill is measured as the benchmark's 2,048-token prompt unless noted (Flash-Next: long prompts through the server). Ornith's speed depends on the chunk size: each prompt chunk streams all 38.9 GiB of experts over
PCIe once (about 3.3 s at 13 GB/s), so a 512-token test reads 150 tok/s and a 2,048-token chunk about 620; larger chunks give more but leave fewer experts resident for decode. Skipping experts does not change prefill
(prompt chunks above the hybrid row limit take the streaming path). The 3,072-token prompt is slower on every model (Bolt about 797, Super-VL about 290, Ornith about 470) for a reason not found.

The two large steps came from (1) sending part of the cache-missed experts over PCIe while the CPU works on the rest (Ornith, `--pcie-frac`, already in the fork engine but off by default) and
(2) not computing cache-missed experts whose routing weight is small (new here: `STRATA_Q36_SKIP_THR`, `STRATA_NH_SKIP_REL`; quality measured as perplexity on forced text, see the expert-skipping section).
Both trade a small, measured quality cost for speed on the CPU-bound models; Flash-Next does not need them.

## Performance over context (2026-11-06)

Decode and prefill against the number of prompt tokens already in context, one or two runs per point, real repository text. Settings per model are in the
model files. Only Flash-Next holds its speed; the others lose decode as the prompt grows, Super-VL least. "-" = not measured (the 35B at a 64k setting stops at 32k, Ornith and Super-VL at 131k).

![Decode speed over context](docs/img/context-decode.svg)

![Prefill speed over context](docs/img/context-prefill.svg)

Decode, tok/s (Flash-Next's 19k and 227k points sit under 16k and 256k):

| Model | 2k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|
| Flash-Next IQ3_XXS | 36.9 | 39.9 | 42.7 | 48.9 | 49.1 | 38.9 | 43.5 |
| Qwen3.6-35B-A3B Q4_K_XL, 262k setting | 40.3 | 34.6 | 31.0 | 29.1 | 17.9 | 11.4 | 6.5 |
| Qwen3.6-35B-A3B Q4_K_XL, 64k setting | 51.3 | 41.0 | 34.2 | 31.1 | - | - | - |
| Ornith 122B IQ3_XXS | 13.8 | 11.8 | 10.8 | 9.5 | 6.5 | 4.3 | - |
| Bolt 74B 24GB QAT | 40.9 | 35.8 | 35.6 | 31.5 | 25.5 | 20.2 | 9.7 |
| Super-VL 120B IQ2_XXS | 7.6 | - | 7.8 | 7.8 | 7.8 | 6.4 | - |

Prefill, tok/s (average over the whole prompt):

| Model | 2k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|
| Flash-Next IQ3_XXS | 383 | 982 | 998 | 1,175 | 1,298 | 1,270 | 1,224 |
| Qwen3.6-35B-A3B Q4_K_XL, 262k setting | 1,168 | 1,337 | 1,344 | 1,306 | 947 | 539 | 229 |
| Qwen3.6-35B-A3B Q4_K_XL, 64k setting | 1,124 | 1,324 | 1,342 | 1,307 | - | - | - |
| Ornith 122B IQ3_XXS | 619 | 632 | 632 | 592 | 395 | 177 | - |
| Bolt 74B 24GB QAT | 926 | 900 | 857 | 763 | 556 | 207 | 74 |
| Super-VL 120B IQ2_XXS | 350 | - | 346 | 328 | 285 | 153 | - |

Points with a caveat: Flash-Next's 2k prefill includes warm-up; Bolt's 131k decode is inflated by a repetition loop in the generated text and its 262k point is one pass;
Ornith's chunk-4096 run and the Super-VL 8k point failed and were queued for a rerun; Flint (llama.cpp) failed at depth and the 35B IQ3_XXS needs a rebuilt engine and is queued.
Per-model tables: [Flash-Next](docs/models/flash-next.md#performance-over-context-2026-11-06), [35B](docs/models/qwen3.6-35b-a3b.md#performance-over-context-2026-11-06),
[Ornith](docs/models/ornith-122b.md#performance-over-context-2026-11-06), [Bolt](docs/models/bolt-74b.md#performance-over-context-2026-11-06),
[Super-VL](docs/models/super-vl-120b.md#performance-over-context-2026-11-06). The data and the plot script are in `docs/bench/`.

## In-depth findings

| Topic | File |
|---|---|
| Flash-Next (production config, tuning sweeps, DFlash2, full-context test) | [docs/models/flash-next.md](docs/models/flash-next.md) |
| Qwen3.6-35B-A3B (DFlash2 port) | [docs/models/qwen3.6-35b-a3b.md](docs/models/qwen3.6-35b-a3b.md) |
| Ornith 122B (PCIe split, profile, prefill chunk, skipping) | [docs/models/ornith-122b.md](docs/models/ornith-122b.md) |
| Bolt 74B (native NVFP4, DFlash, prefetch, skipping) | [docs/models/bolt-74b.md](docs/models/bolt-74b.md) |
| Super-VL 120B | [docs/models/super-vl-120b.md](docs/models/super-vl-120b.md) |
| Flint 40B (llama.cpp) | [docs/models/flint-40b.md](docs/models/flint-40b.md) |
| Expert skipping, all models | [docs/expert-skipping.md](docs/expert-skipping.md) |
| KV-cache codec quality test | [docs/kv-codecs.md](docs/kv-codecs.md) |
| Method, noise levels, shared findings | [docs/tuning-method.md](docs/tuning-method.md) |
| Why these model sizes | [docs/models/README.md](docs/models/README.md) |

## Not here

The llama.cpp work for this card lives in [`sterlenjohnson/llama.cpp`](https://github.com/sterlenjohnson/llama.cpp)
branch `cuda-ada-4060ti`. No Strata measurements are recorded there.
