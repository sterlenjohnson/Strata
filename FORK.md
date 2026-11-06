# Fork notes: Strata on an RTX 4060 Ti 16 GB

Branch `cuda-ada-4060ti`, based on upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata) v0.2.10 (`ffa80d73`). This branch carries notes only; the engine changes
(Ornith, Bolt/Super-VL, expert skipping, 4060 Ti tuning) live on a separate branch. In-depth findings are one file per model in [`docs/models/`](docs/models/README.md).

**Machine:** RTX 4060 Ti 16 GB (PCIe 4.0 x8, 13 GB/s measured, **120 W limit**), Ryzen 7 3700X as a 6-core / 12-vCPU Fedora VM on Proxmox, 60 GB RAM (about 47 GB usable as an expert arena).
Models are about 47 GB of non-PLE weights, so dense weights sit in VRAM and experts split between VRAM and RAM ([why](docs/models/README.md)).

## Speed by context length

Peak speed at 2k tokens flatters most models: **only Flash-Next holds its speed as the prompt grows**. The others lose decode speed with context (Bolt halves by 64k, Ornith by 64k, the 35B by 64k at a 262k setting) and prefill collapses beyond 64k.
Real text, one or two runs per point, best settings per model, x = prompt tokens already in context. The graphs show one line per model; setting variants (YaRN 2, the 35B at a 64k setting, Ornith chunk 4096) are in the tables and the model files.

![Decode speed over context](docs/img/context-decode.svg)

![Prefill speed over context](docs/img/context-prefill.svg)

**Decode, tok/s**

<!-- decode -->
| Model | 2k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash-Next IQ3_XXS | 36.9 | 39.9 | 42.7 | 48.9 | 49.1 | 38.9 | 43.5 | - | - |
| Flash-Next, YaRN 2 (512k) | - | - | - | - | - | - | 38.1 | - | - |
| Flash-Next, YaRN 4 (1M) | - | - | - | 34.5 | - | - | 31.6 | - | - |
| 35B Q4_K_XL (262k setting) | 40.3 | 34.6 | 31.0 | 29.1 | 17.9 | 11.4 | 6.5 | - | - |
| 35B Q4_K_XL (64k setting) | 51.3 | 41.0 | 34.2 | 31.1 | - | - | - | - | - |
| 35B IQ3_XXS (262k setting) | 73.1 | 60.1 | 48.3 | 34.7 | 22.0 | 12.8 | 6.9 | - | - |
| Ornith 122B IQ3_XXS | 13.8 | 11.8 | 10.8 | 9.5 | 6.5 | 4.3 | - | - | - |
| Bolt 74B QAT | 40.9 | 35.8 | 35.6 | 31.5 | 25.5 | 20.2 | 9.7 | - | - |
| Super-VL 120B IQ2_XXS | 7.6 | 7.5 | 7.8 | 7.8 | 7.8 | 6.4 | - | - | - |
| Ornith, `--chunk 4096` | 13.1 | 11.5 | 10.2 | 8.7 | - | - | - | - | - |
| Flint 40B Q8_0 (llama.cpp, pp512) | - | 29.7 | 28.6 | 28.1 | 24.9 | 20.7 | - | - | - |
<!-- /decode -->

**Prefill, tok/s** (average over the whole prompt)

<!-- prefill -->
| Model | 2k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash-Next IQ3_XXS | 383 | 982 | 998 | 1,175 | 1,298 | 1,270 | 1,224 | - | - |
| Flash-Next, YaRN 2 (512k) | - | - | - | - | - | - | 1,133 | - | - |
| Flash-Next, YaRN 4 (1M) | - | - | - | 903 | - | - | 983 | - | - |
| 35B Q4_K_XL (262k setting) | 1,168 | 1,337 | 1,344 | 1,306 | 947 | 539 | 229 | - | - |
| 35B Q4_K_XL (64k setting) | 1,124 | 1,324 | 1,342 | 1,307 | - | - | - | - | - |
| 35B IQ3_XXS (262k setting) | 1,254 | 974 | 944 | 884 | 702 | 452 | 211 | - | - |
| Ornith 122B IQ3_XXS | 619 | 632 | 632 | 592 | 395 | 177 | - | - | - |
| Bolt 74B QAT | 926 | 900 | 857 | 763 | 556 | 207 | 74 | - | - |
| Super-VL 120B IQ2_XXS | 350 | 350 | 346 | 328 | 285 | 153 | - | - | - |
| Ornith, `--chunk 4096` | 629 | 1,094 | 993 | 797 | - | - | - | - | - |
| Flint 40B Q8_0 (llama.cpp, pp512) | - | 209 | 207 | 200 | 188 | 163 | - | - | - |
<!-- /prefill -->

Flash-Next's 19k and 227k points sit under 16k and 256k; its 2k prefill includes warm-up. Bolt's 131k decode is inflated by a repetition loop and its 262k point is one pass. "-" = not measured. Flint (llama.cpp) prefill is one 512-token chunk at that depth, not a whole-prompt average.

## Start against now (short context)

| Model | Decode, start to now | Prefill now | What did it | Details |
|---|---|---|---|---|
| Flash-Next IQ3_XXS (production) | 41 to 41 | ~1,200 | nothing beat the production config | [file](docs/models/flash-next.md) |
| Bolt 74B 24GB QAT | 18 to **37** | ~920 | native NVFP4, expert skipping (`STRATA_NH_SKIP_REL=0.5`), DFlash depth 3 | [file](docs/models/bolt-74b.md) |
| Bolt 74B UD-Q3_K_XL | 15 to 23 | ~500 | DFlash depth 3, prefetch 12 | [file](docs/models/bolt-74b.md) |
| Super-VL 120B IQ2_XXS | 5 to **8.3** | ~350 | 8 threads, prefetch 16, expert skipping | [file](docs/models/super-vl-120b.md) |
| Ornith 122B IQ3_XXS | 3.3 to **17** (27 for code) | ~620 (~1,040 at `--chunk 4096`) | `--pcie-frac 0.5`, expert skipping (`STRATA_Q36_SKIP_THR=0.10`), q8 KV | [file](docs/models/ornith-122b.md) |
| Qwen3.6-35B-A3B Q4_K_XL / IQ3_XXS | n/a to 40-59 / **73** | ~1,200-1,340 / ~1,250 | hybrid experts, q8 KV; IQ3_XXS keeps 184 of 256 experts in VRAM; a smaller `--max-context` keeps more resident | [file](docs/models/qwen3.6-35b-a3b.md) |
| Flint 40B Q8_0 (llama.cpp) | n/a to 32 | 228 (691 at `-ub 2048`) | `-ncmoe 33 -t 6 -ub 2048` | [file](docs/models/flint-40b.md) |

Two changes gave the big steps: sending part of the cache-missed experts over PCIe while the CPU runs the rest (`--pcie-frac`), and skipping cache-missed experts whose routing weight is small.
Skipping trades a measured quality cost for speed (Ornith +3% prose / +2% docs perplexity at 0.10, about 0 on code; Bolt about 0; see [expert-skipping.md](docs/expert-skipping.md)). Start and now were not always the same prompts or build.
The two tables use different prompts, so the same model reads differently (Ornith 17 on a prose prompt at a 16k setting, 13.8 on repository text in the context sweep; Bolt 37 against 40.9). Not yet measured across the full range: Flint (see its file).

## More

| | |
|---|---|
| [docs/models/](docs/models/README.md) | one file per model: flags, sweeps, negative results |
| [docs/expert-skipping.md](docs/expert-skipping.md) | skipping rules and quality checks |
| [docs/kv-codecs.md](docs/kv-codecs.md) | KV-cache codec quality test |
| [docs/tuning-method.md](docs/tuning-method.md) | how numbers were taken, noise levels |
| [docs/bench/](docs/bench/) | context data and the plot script |

The llama.cpp work for this card is in [`sterlenjohnson/llama.cpp`](https://github.com/sterlenjohnson/llama.cpp) branch `cuda-ada-4060ti`; no Strata measurements are recorded there.
