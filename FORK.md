# Fork notes: Strata on an RTX 4060 Ti 16 GB (Fedora VM on Proxmox)

Branch `cuda-ada-4060ti`, based on upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata) `main` at v0.2.10
(`ffa80d73`). **Status: notes only, no code changes yet.** The fork-only work (39 commits: tool-call recovery in
`serve`, Nemotron-H/Bolt, Ornith, 4060 Ti tuning, a measurement harness) still lives on a separate branch and has not
been ported onto v0.2.10. Numbers below are measurements, with the engine and log they came from.

## Test machine

| | |
|---|---|
| GPU | RTX 4060 Ti 16 GB (288 GB/s, PCIe 4.0 x8, 13.0 GB/s host-to-device measured), sm_89 |
| CPU / RAM | Ryzen 7 3700X (6 cores presented as 12 vCPUs), 60 GB RAM (about 47 GB usable as an expert arena) |
| OS | Fedora 44 **virtual machine** on **Proxmox**. KSM on the host hurt performance, so numbers may differ on bare metal; record the host's KSM state with each run |
| Build | upstream v0.2.10 built in a CUDA 12.8 container (`-j2`, 40 GB cap); the host nvcc cannot compile against GCC 16 |

## Where things stand (2026-11-06)

Decode tok/s on this box (RTX 4060 Ti 16 GB at a 120 W limit, 6-core Zen 2 VM), start of the tuning work against now. Details and the logs behind each number are in the sections below.

| Model | Start | Now | What did it |
|---|---|---|---|
| Flash-Next IQ3_XXS (production) | about 41 | about 41 | unchanged: every flag, the DFlash2 drafter, a weight-skip port and `--pcie-frac` 0.55-0.65 left it within noise |
| Bolt 74B 24GB QAT | 18 | **about 37** | `--nvfp4 native` (33.5), `STRATA_NH_SKIP_REL=0.5` (37), DFlash depth 3, 8 threads, prefetch 12 |
| Bolt 74B UD-Q3_K_XL | 15 | about 23 | DFlash depth 3, prefetch 12 |
| Super-VL 120B IQ2_XXS | 5 | **about 8.3** | 8 threads, prefetch 16, `STRATA_NH_SKIP_REL=0.5-0.7` |
| Ornith 122B IQ3_XXS | 3.3 | **about 17 (up to 27 for code)** | `--pcie-frac 0.5`, 7 threads, q8 KV, 16k context, `STRATA_Q36_SKIP_THR=0.10` (0.12-0.15 for code) |
| Flint 40B Q8_0 (llama.cpp) | not run | about 32 | `-ncmoe 33 -t 6 -ub 2048` |

The two large steps came from (1) sending part of the cache-missed experts over PCIe while the CPU works on the rest (Ornith, `--pcie-frac`, already in the fork engine but off by default) and
(2) not computing cache-missed experts whose routing weight is small (new here: `STRATA_Q36_SKIP_THR`, `STRATA_NH_SKIP_REL`; quality measured as perplexity on forced text, see the expert-skipping section).
Both trade a small, measured quality cost for speed on the CPU-bound models; Flash-Next does not need them.

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

## DFlash2 on Qwen3.6-35B-A3B (new in this branch)

Upstream v0.2.10 runs the DFlash2 drafter only on Flash-Next. This branch runs the Qwen3.6-35B-A3B drafter
(`qwen3.6-35b-a3b-dflash2.q8.pack`) on the 35B itself (`strata --model <35B GGUF> --draft dflash2 --draft-model <pack> --spec 5`).

What changed:
- **Drafter shapes** (`src/core/dflash2.cpp`): the loader took only hidden size 2560. It now follows the pack, and two buffers that were
  sized for Flash-Next are fixed: the quantized-tap buffers held 5 taps (the 35B has 8, so the context kernels overflowed them) and the
  convolution coefficients were laid out 640 wide with the second half at +320 (that width is hidden/4: 512 for the 35B). Flash-Next's
  2560 gives the same values as before.
- **Engine hooks** (`src/qwen35/q36_engine.cu`): taps are copied out of the residual stream after the pack's 8 layers, in the prompt
  chunks, single steps and verify windows; the drafter reads the engine's own embedding and output matrix (no copy); a DFlash2 step
  drafts, verifies the window and commits like the MTP step. `STRATA_Q36_DF2_TAP_SHIFT` moves the tap layers (0 = after the layer the pack
  names); `STRATA_Q36_DF2_TRACE=1` prints drafts against picks.

Measured (RTX 4060 Ti VM, UD-Q4_K_XL, 200 tokens, temperature 0, one run each, no KV quantization, 8 CPU threads):

| Prompt | Plain decode | DFlash2 depth 5 | Drafts accepted |
|---|---|---|---|
| code (LRU cache) | 55.6 tok/s | 90.7 | 73% |
| prose (immune system) | 57.7 | 63.2 | 49% |
| JSON list | 51.7 | 56.9 | 45% |

Q8_0 file (`Qwen3.6-35B-A3B-Q8_0.gguf`, no MTP block, 35 GB so most experts sit in host memory), same prompts, depth 5:

| Prompt | Plain decode | DFlash2 | Drafts accepted |
|---|---|---|---|
| code | 32.4 tok/s | 38.9 (+20%) | 67% |
| prose | 32.9 | 37.4 (+14%) | 54% |
| JSON | 32.4 | 34.4 (+6%) | 49% |

Q8_0 depth sweep (`--spec N`, 200 tokens, temperature 0, one run per cell; plain decode 32.4 code / 32.9 prose):

| Depth | Code tok/s | Code accepted | Prose tok/s | Prose accepted |
|---|---|---|---|---|
| 2 | 41.5 | 82% | 41.0 | 81% |
| 3 | 44.9 | 82% | 34.5 | 66% |
| 4 | 47.1 | 73% | 38.1 | 59% |
| 5 | 38.9 | 67% | 37.4 | 54% |
| 6 | 44.3 | 63% | 39.2 | 48% |
| 7 | 37.1 | 59% | 29.8 | 44% |

Acceptance falls steadily with depth; throughput does not move smoothly (differences under about 5 tok/s are within the run-to-run
noise seen here, and depth 5 was not rerun). Depth 2-4 is the useful range: 4 for code, 2 for prose. Every prose run matched plain
decoding token for token; all code runs matched each other but differ from plain decoding.

The Q8_0 file needed one loader change: `nextn_predict_layers` is now optional in the architecture guard (a file without it has no MTP
block). DFlash2 is an external pack, so it does not depend on that block. The prose answer matched plain decoding token for token.

Code prompt at other settings: DFlash2 depth 3 reached 80.0 (84% accepted), depth 7 reached 88.4 (63%), and the model's own MTP at depth 2
reached 80.5 (83%). Before the buffer fixes the same pipeline accepted 2 of 625 drafts.

Caveats: DFlash2 output matches plain decoding for the first 111-137 tokens of each 200-token answer and then differs; the MTP at depth 2
matches exactly. The divergence is late, which fits near-tied tokens under different window kernels, but it is not proven to be that. The
Ornith drafter (hidden 3072, block size 16) loads but is not run: its block size needs 16 rows, the drafter uses 8. The path is
single-request (`--batch 1`); the serve loop is wired but untested, and there is no prompt-reuse handling (`begin_request`) yet.

## DFlash2 on Ornith 1.5 122B (runs, but slower)

The upstream-based build cannot load the Ornith file: its layer 0 experts are IQ1_S/IQ2_S, which the native expert kernels there do not take.
The drafter was therefore added to the fork's engine (branch `ornith-dflash2` in the fork's repository, which already runs Ornith), keeping the
same hooks. The drafter's block is 16 rows; the engine uses the first 8, so drafts are weaker than the pack was trained for.

Ornith UD-IQ3_XXS, code prompt, 100 tokens, q8 KV, 8 threads, one run each (plain decode 8.9 tok/s code, 8.4 prose):

| DFlash2 depth | Decode tok/s | Drafts accepted |
|---|---|---|
| 1 | 8.0 | 85% |
| 2 | 7.5 | 65% |
| 3 | 6.7 | 56% |
| 4 | 5.3 | 38% (prose: 5.2, 34%) |

Every depth is slower than plain decoding. Most of Ornith's experts sit in host memory, so a verify window of several rows reads several
rows' worth of experts per step, and the drafter and its buffers take about 1.2 GiB of VRAM from the expert slots. The 8-row
truncation may also be costing acceptance (Flash-Next reaches 65% and the 35B 45-73% with their native 8-row blocks). Not tried: the full
16-row block, fewer taps, or a profile that gives the freed VRAM back to experts.

## KV-cache compression codecs (quality test)

`--kv-turbo k4v3 --kv-rot wht|planar|iso|ggml` (Qwen3.6-35B-A3B engine) rounds every K and V row through a codec as it is written and
writes the decoded values back. The cache stays fp32, so this measures quality, not memory; packed storage is not built.
Codecs: TurboQuant (norm, sign-flipped Walsh-Hadamard rotation over 128-value groups, Lloyd-Max centroids, norm corrected by the chosen
centroids' energy; format from TheTom/llama-cpp-turboquant), RotorQuant's PlanarQuant (random 2D rotation per pair) and IsoQuant (random
quaternion per 4-block), no norm correction as in its reference, and ggml's q8_0 / q4_0. The rotation parameters are seeded from this
engine's own random stream, not the reference's.

Perplexity of 1,500 teacher-forced tokens of technical prose after a 1,500-token prompt (UD-Q4_K_XL, one run each). fp32 cache: 10.512.
The ggml runs landed below it, so differences of about 0.04 are noise. Bytes are for K + V of one cell in one attention layer (512 values
each), calculated from block sizes (turbo4 66 B, turbo3 50 B, turbo2 about 34 B per 128 values; q8_0 34 B, q4_0 18 B per 32).

| K / V | Bytes | ggml | TurboQuant | PlanarQuant | IsoQuant |
|---|---|---|---|---|---|
| fp32 | 4096 | 10.512 | | | |
| 8 / 8 | 1088 | 10.491 | | | |
| 8 / 4 | 832 | 10.468 | | | |
| 4 / 4 | 528-576 | 10.505 | 10.515 | 10.781 | 10.798 |
| 4 / 3 | 464 | | 10.548 | 11.198 | 10.937 |
| 3 / 3 | 400 | | 10.658 | 11.251 | 10.920 |
| 3 / 2 | 336 | | 10.760 | 11.522 | 11.567 |
| 2 / 2 | about 272 | | 10.964 | 11.984 | 11.483 |

Reading: at 4 bits ggml q4_0 matches TurboQuant; TurboQuant's value is below 4 bits (K4/V3 +0.3% at 464 B, K3/V3 +1.4% at 400 B, K3/V2
+2.4%). On this model RotorQuant's rotations are worse than the Walsh-Hadamard one at every width. Norm correction changes little
(planar K3/V3 11.251 -> 11.121, iso 10.920 -> 10.925, TurboQuant K3/V2 10.760 -> 10.795 without it). One model, one text, approximate
tokenization; the rotations are applied per 128-value group (the head dim is 256), and the speed advantage claimed for RotorQuant is not
measured here.

With the drafters (200 tokens, depth 4 DFlash2 / depth 2 MTP): DFlash2 on the code prompt 90.3 tok/s uncompressed, 87.7 with K4/V3,
88.2 with K3/V3 (drafts accepted 73% / 69% / 74%); prose 77.8 / 74.2 / 68.7 (54% / 52% / 57%). MTP code 71.8 -> 76.5 (83% -> 86%), prose
84.4 -> 80.9 (85% -> 77%). All within the run-to-run noise: compressed KV does not break drafting. The DFlash2 drafter's own ring is
separate and untouched.

## Why test models of about 47 GB

Flash-Next IQ3_XXS is a 75.96 GB file, but 28.80 GB of it is the PLE table (IQ4_NL, trigram-addressed). Decode reads about
16 rows of 90 B from it per token, so it sits on the SSD and costs almost nothing in RAM or VRAM. What the box actually
has to hold is the remaining **47.2 GB**: the dense weights in VRAM and the routed experts split between VRAM slots and
a RAM arena. That matches the machine, which has 16 GB of VRAM and about 47 GB of RAM usable as an expert arena.

So the useful comparison set is models whose non-PLE weights are in the same 40-51 GB range. They fit the same design
(dense in VRAM, experts across VRAM and RAM), which means the same levers apply: a routing profile, prefetch, a drafter,
quantized KV.

| Model | Non-PLE weights | Routed experts | Active per token |
|---|---|---|---|
| Flash-Next IQ3_XXS | 47.2 GB | 42.9 GB | 10 of 512 experts |
| Super-VL 120B UD-IQ2_XXS | 51.1 GB | 41.3 GB | 22 of 512 experts (12B active) |
| Ornith 1.5 122B UD-IQ3_XXS | 49.5 GB | 41.6 GB | 8 of 256 experts |
| Flint 40B Q8_0 | 42.3 GB | 40.2 GB | 6 of 256 experts (never loaded) |
| Bolt 74B UD-Q3_K_XL | 39.0 GB | 32.1 GB | 9 of 240-496 experts |
| Bolt 74B 24GB QAT | 22.0 GB | 17.1 GB | 9 of 240-496 experts |

Sizes come from GGUF headers (see the Flash-Next vs Bolt/Ornith research report). The 47 GB figure is the file size minus
the PLE table; it is arithmetic on those numbers, not a separate measurement.

Models that look like candidates but are **not** confirmed here: Nemotron 3.5 Super (the Super-VL above is its vision
variant), Kolibri-1 (78B total, 3.46B active, so smaller than this range), and gpt-oss-120b (117B total, 5.1B active, but
about 63 GB in its native MXFP4 form, over the arena). A web search found no other recent model of this size with a
confirmed GGUF, so check Hugging Face directly before downloading anything.

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

## Tuning sweeps, part 3: Ornith profile, threads, prefill chunk (2026-11-05)

Same box, GPU power limit 120 W. Ornith 122B IQ3_XXS on the fork engine (16k context, q8 KV, no MTP block, `--pcie-frac 0.5`, 200 generated tokens of a prose
prompt unless noted), warm decode in tok/s. Differences under about 0.5 tok/s are noise; rows are one run each unless a count is given.

**Where the time goes (`STRATA_Q36_PROF=1`).** In a 200-token run (9,600 decode layers, 16.2 s) the CPU expert pool is busy 12.1 s: gate/up 9.4 s, down 2.7 s,
activation quantization 0.03 s, 97.2 GB streamed (8 GB/s on average). Per layer the GPU front half costs 0.07 ms to enqueue and 0.48 ms to wait for the routing,
and the hybrid MoE step 1.56 ms (1.77 ms in a second profile run), about two thirds of the layer. The remaining decode time is the expert phase, not launches.

| Setting | Decode tok/s |
|---|---|
| Base, 7 threads (four runs) | 12.7 / 12.6 / 12.7 / 12.1 |
| Base, 5 threads | 11.9 |
| 5 threads pinned to even / odd / first-six logical CPUs | 12.2 / 11.6 / 11.7 |
| `STRATA_Q36_THP=1` (huge-page arena), 5 / 7 threads | 12.4 / 12.9 |
| THP + even CPUs, 5 threads | 12.8 |
| `STRATA_IQ_PREFETCH` 0 / 1024 / 4096 / 8192 | 12.4 / 12.1 / 12.5 / 12.2 |
| `STRATA_IQ256_GATHER=1` / `CUDA_DEVICE_MAX_CONNECTIONS=32` | 12.3 / 12.2 |

Nothing moves the number outside the 12.1-12.7 spread of the base runs. THP is higher in all three pairs (+0.2 to +0.6) but within that spread, so it is not
adopted. Pinning to the even or odd logical CPUs does not help; the VM's 12 logical CPUs are 6 cores.

**Staging buffers.** `STRATA_Q36_STAGE_BUFS=1` (one PCIe staging buffer instead of two) leaves decode unchanged (11.1 against 11.0-11.3) and cuts prefill from
598-634 to 484-514 tok/s at 2,048 tokens (-19%, two runs each). Two buffers stay.

**Prefill chunk at an 8k prompt.** `--chunk 4096` prefills at 1,034-1,043 tok/s against 624-634 for the default 2048 (+64%), but the larger staging area leaves 6
experts per layer resident instead of 16 and decode falls from 10.4-11.0 to 10.0-10.3. `--chunk 8192` is refused (`bad --chunk / --max-context`). Use 4096 only for
prompt-heavy work.

**Next-layer routing guess (`STRATA_Q36_PREDICT=1`, accuracy only).** The host guess one layer ahead matches 79% of the actual picks and 77% of the picks that
have no slot.

**Bolt UD-Q3_K_XL (IQ3 file), DFlash depth 3, 8 threads, 300 MiB reserve.** The old setting (6 threads, prefetch 4) reads 15.1-15.6. With DFlash and prefetch 16 it reads
21.1-21.6 (two runs) and prefetch 12 reads 22.2-24.0 (one run). Without DFlash, 8 threads with prefetch 12 or 16 read 13.8-14.4, below the old setting, so for this file the
gain is the drafter alone. Prefill is 499-501 tok/s (441-505 before).

## Expert skipping: drop low-weight experts that would run on the CPU (2026-11-06)

Experts with no VRAM slot cost a CPU pass (or a PCIe copy) each, and many of them carry a small routing weight. Skipping them removes that work. Only
CPU-side entries are ever skipped; experts resident in VRAM or already prefetched are always computed. Off by default and bit-exact when unset.

| Engine | Env var | Rule |
|---|---|---|
| Ornith (`strata-q36`) | `STRATA_Q36_SKIP_THR=w` | skip a slot-less expert whose routing weight is below w (weights sum to 1 over the top 8) |
| Super-VL, Bolt (`strata-nh`) | `STRATA_NH_SKIP_REL=f` (or `STRATA_NH_SKIP_THR=w`) | skip below f times the row's largest weight (absolute weights depend on top-k and the model's scale, so use REL) |
| Flash-Next (`strata`) | `STRATA_SKIP_THR=w` | same rule as Ornith; `STRATA_SKIP_HIST=1` prints the weight quantiles; works beside the upstream rank-based `STRATA_ROUTE_TAIL_SKIP` |

Also available: `STRATA_Q36_SKIP_RENORM=1` / `STRATA_NH_SKIP_RENORM=1` (rescale the kept weights; worse, see below), `STRATA_Q36_NLL=1` / `STRATA_NH_NLL=1` with `--tokens PROMPT --force IDS`
(teacher-forced perplexity through the skipping path; the output line `expert skipping (...): N of M CPU-side entries skipped` gives the skip rate).

**Ornith 122B** (16k context, q8 KV, no MTP block, `--pcie-frac 0.5`, 7 threads). Speed: 200 tokens of a prose prompt, base 12.1 tok/s (code prompt base 12.1 too in this test).
Perplexity change: 600 forced tokens after a 1,500-token prompt, one text per type; base perplexity prose 8.222, Python 7.816, TypeScript 3.255, docs 8.159. One run per cell
unless "x2".

| w | CPU-side entries skipped | Decode tok/s (prose / code) | Perplexity prose | Python | TypeScript | Docs |
|---|---|---|---|---|---|---|
| 0.04 | 2% | 12.4 | +0.8% | - | - | - |
| 0.07 | 22-24% | 13.0 | +1.2% | +0.9% | - | - |
| 0.085 | 43% | 14.6 / 15.2 | +3.1% | - | - | - |
| 0.10 | 62-66% | 17.2, 17.1 (x2) / 18.0 | +3.4% | -2.0% | -0.4% | +2.0% |
| 0.12 | 81-84% | 21.6 | +7.8% | -3.4% | - | - |
| 0.15 | 93-96% | 27.0 | +12.2% | -3.7% | +2.5% | +14.8% |
| 0.20 | 99% | - | - | - | +7.9% | +51.9% |

Python and TypeScript stay at or better than base up to 0.12-0.15 (the 600-token samples are small, so a 2-4% gain is within sampling noise, but none of them degrades). Prose and docs
pay about 3% at 0.10 and degrade fast beyond. **Recommendation: `STRATA_Q36_SKIP_THR=0.10`** (+42% decode, 3% perplexity on prose); up to 0.12-0.15 for code-only work; 0.07
(+7%, +1.2%) when quality matters most. Top-1 agreement moves by under 1.5 points at 0.10.

Variants that did not help: renormalising the kept weights is worse (perplexity +3.9% at 0.085, +14.0% at 0.10, +32.4% at 0.12; same speed as plain). A cap on the skipped weight mass per token
(`mass cap`) is no better than the plain threshold: cap 0.25 gives 17.1 tok/s at +4.9% against 17.2 at +3.4% for w = 0.10. Deferring the same experts until after the next layer's
attention (`STRATA_Q36_DEFER`) gave 12.0-12.2 tok/s (no gain at `--pcie-frac 0.5`) and +0.0% / +0.5% / +2.0% perplexity for 0.07 / 0.10 / 1.0. Next-layer prefetch gave no gain either (above).

**Super-VL 120B** (`STRATA_NH_SKIP_REL`, 8 threads, prefetch 16, 600 forced tokens, one run each): perplexity base 17.43; REL 0.15 / 0.25 / 0.35 / 0.5 / 0.7 gives 17.25 / 16.67 / 16.52 / 16.31 / 16.40
(slightly better than base, one 600-token text) with 7% / 22% / 45% / 74% / 92% of CPU-side entries skipped. Decode (tg64): base 6.6-6.9, REL 0.25 7.0, 0.35 7.5, 0.5 8.1 (+18%), 0.7 8.5 (+23-25%). Recommended: 0.5.
This model's 22-of-512 routing carries many small weights, which is why it tolerates so much.

**Bolt 74B 24GB QAT** (`STRATA_NH_SKIP_REL` 0.15 / 0.25 / 0.35 / 0.5, DFlash depth 3): perplexity 20.52 base against 20.52 / 20.65 / 20.47 / 20.68 (unchanged), 2-47% skipped, decode 27.0 / 29.5 / 29.6 / 28.4
against 28.5 base: no speed gain (noise is about 2 tok/s). Bolt is bound by its dense weights and the drafter, not by the expert tail, so leave it off.

**Flash-Next** (`STRATA_SKIP_THR`, production config, mean of the three short prompts, one run each; plain runs 39.1 and 38.6): 0.03 38.9, 0.05 37.3, 0.07 39.6, 0.05 renormalised 37.0.
No gain: only 5-15% of lookups miss and the upstream rank-based tail skip (`STRATA_ROUTE_TAIL_SKIP=7`, on by default) already takes the cheap ones. With the tail skip switched off
(`STRATA_ROUTE_TAIL_SKIP=0`) decode falls to 29.8, and adding `STRATA_SKIP_THR=0.05` brings it back only to 31.4, so keep the upstream rule and leave `STRATA_SKIP_THR` unset.

**How to pick a value.** Start with the table, run `STRATA_Q36_NLL=1` on a text like your workload (`--tokens` prompt, `--force` continuation), and take the largest threshold whose perplexity
stays within about 3% of w = 0. Check the skip-rate line: a rate above 90% means the model is running almost entirely on the VRAM-resident experts.

**Does skipping hurt generated code?** Four coding prompts (binary search, word frequency, merge intervals, LRU cache; "reply with only the code block, with 3 asserts"), up to 1,400 tokens
at the default sampling of the engine's `--tokens` path (greedy), each extracted and run in Python: base 3 of 4 pass (the LRU answer's own asserts fail), w = 0.10 3 of 4 (the merge-intervals function is
correct but one of the model's asserts is wrong), w = 0.15 3 of 4 (the LRU answer is cut inside the reasoning block). Equal pass rates on four prompts is not a significant result, but there is no sign of a code regression
and decode was 1.9x faster at 0.10 and 3.5x at 0.15 on the long generations.

**Files changed** (all uncommitted in the working trees):
- `/mnt/AI/Strata-ornith/src/qwen35/q36_engine.cu`: `STRATA_Q36_SKIP_THR`, `SKIP_RENORM`, mass cap, `DEFER`, `NLL`, `PROF`, `PREDICT`.
- `/mnt/AI/Strata-ornith/src/nemotron_h/nh_engine.cu`: `STRATA_NH_SKIP_REL/THR/RENORM`, `STRATA_NH_NLL` (skip decided before the PCIe share, so skipped experts are never uploaded).
- `/mnt/AI/Strata-main`: `src/core/expert_source.cpp` and `include/strata/core/expert_source.hpp` (decision and counters), `src/core/verify.cpp` and `include/strata/core/verify.hpp`
  (weights to the host, renormalised combine), `src/kernels/cuda/elementwise.cu` and `include/strata/kernels/elementwise.hpp` (doorbell publishes weights), `src/program/generate.cpp` (stats line).

## Bolt 74B 24GB QAT: native NVFP4 weights (2026-11-06)

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

## Flint 40B (arch `flint`, Q8_0, 42 GB): runs in llama.cpp, not in Strata (2026-11-06)

Strata has no `flint` architecture. The prebuilt llama.cpp b11798 at `/mnt/AI/llcpp-11798/root` loads it (the box has CUDA 12.6 only, so the CUDA 13 runtime libraries were extracted from the NVIDIA pip wheels
into the scratchpad's `cu13/lib`). `llama-bench -ngl 99 -fa 1 -t 6 -ub 512 -p 512 -n 128 -r 2`, decode / prefill in tok/s: `-ncmoe` 40: 29.1 / 190; 38: 28.7 / 204; 36: 30.8 / 203; 35: 31.2 / 219; 34: 31.4 / 214; **33: 32.1 / 228**
(`-ncmoe 28` does not fit: 18.8 GiB of experts on the 16 GB card; 32 and below were not tried). At `-ncmoe 36`: 4 / 5 / 6 / 7 / 8 threads give 30.8 / 30.5 / 30.8 / 30.2 / 28.5. Larger micro-batches
help prefill a lot: pp2048 at `-ub` 1024 / 2048 reads 368 / 691 tok/s. `llama-server --n-cpu-moe 38 -c 8192` produced 200 tokens at about 29 tok/s. Use `-ncmoe 33 -t 6 -ub 2048`. Flint was only run for speed here
(the file is a "Preview" base model; the quick completion test continued the prompt as plain text).

## Flash-Next: nothing further (2026-11-06)

`--pcie-frac` 0.55 / 0.60 / 0.65 added to the production config (two runs each, three short prompts) read 38.7-42.8 against 37.7-43.4 for the plain config: no gain. The weight-based skip port (above) and
every earlier flag change also leave it at about 41 tok/s. The production configuration is unchanged.

## Tuning sweeps, part 2: PCIe split, prefetch, prefill (2026-11-05)

Same box (6 cores / 12 threads in the VM, pinned; PCIe 4.0 x8, 13 GB/s measured; **the GPU power limit was 120 W** for every number in this file, against a 165 W default). Warm decode, one run each unless two numbers are
shown; differences under about 0.5 tok/s (Ornith, Super-VL) or 2 tok/s (Bolt) are noise.

**Ornith 122B, `--pcie-frac` (fork engine; 16k context, q8 KV, no MTP block, 7-8 threads).** `--pcie-frac F` sends that fraction of a
batch's slot-less experts over PCIe to the GPU while the CPU pool runs the rest. It defaults to 0 and had not been tried.

| `--pcie-frac` | Decode tok/s (synthetic prompt) |
|---|---|
| 0 | 7.7 / 7.9 |
| 0.25 | 9.8 / 10.6 |
| 0.4 / 0.5 / 0.6 | 10.7 / 10.5-11.2 / 10.7-10.8 (a flat plateau) |
| 0.75 | 10.1 / 9.9 |
| 1.0 (CPU idle) | 8.9 / 9.0 |

On real text the same setting reads 12.6 tok/s (prose) and 6.9 (code; code routes to a wider spread of experts). Threads: 6 / 7 / 8 gave
11.0 / 11.2 / 10.5-11.2, 10 gave 8.2 (the VM has 6 cores). Context 8k and 16k are equal (11.0-11.3). Prefill chunk 512 and 2048 are equal
(about 620 tok/s at 2-4k tokens); chunk 1024 halves it, 256 gives 77, 128 gives 12-29. The 512-token prefill test (150 tok/s) understated
Ornith: each pass has a fixed cost of about 3.3 s, so it reaches 600+ tok/s from 2,048 tokens up.

Next-layer prefetch (`STRATA_Q36_PREFETCH=1`, worktree branch `ornith-dflash2`, off by default): the GPU runs the next layer's router on this
layer's hidden state (77-89% of the guessed slot-less experts are used) and copies them into a second staging buffer one layer ahead. It
gave no speedup (12.0 / 9.7 / 11.7 / 10.0 against 12.6 plain), so the PCIe link and the CPU pool are already close to saturated together.

DFlash2 on Ornith with `--pcie-frac 0.5`: plain 6.9 code / 12.3 prose; depth 1 gave 6.4 / 11.9, depth 2 7.1 / 10.5, depth 3 5.9 / 8.4,
depth 4 3.9 / 7.2. A depth-1 step costs 285 ms against 145 ms for a plain token (1.98x), and yields 1.82 tokens, so it is slower; the drafter
also halves the resident experts (16 to 8 per layer).

**Bolt 24GB QAT (DFlash depth 3, 8 threads, 300 MiB reserve).** Prefetch 12 gave 29.2-30.2, 16 gave 28.8-30.9, 24 gave 26.2-27.9 (the earlier
best, 6 threads with prefetch 8, was 26-27). A later baseline run read 27.3-27.8 at prefetch 16. `--pcie-frac` does nothing at 0.10 (27.3-27.9)
and hurts above that (0.50: 14-18, 0.75: 15-17, with prefill collapsing from 900 to 100-600 tok/s).

**Super-VL 120B.** 8 threads and prefetch 16 gave 6.9 / 7.2 (20 and 24 gave 6.6-6.9). `--pcie-frac` 0.15-0.25 gave 7.1-7.3 (not different);
0.5 and 0.75 are worse (6.7-6.9, 6.2-6.4). Prefill is 345-357 tok/s throughout.

**Prefill at larger prompts** (tok/s): Bolt 900 (2,048 and 4,096 tokens; 797 at 3,072), Super-VL 350 (290 at 3,072), Ornith 620 (470 at 3,072).

**Not measured, suggested:** the GPU power limit reads 120 W against a 165 W default, and the "SW Power Capping" counter shows about 892 s
of capping; raising it in LACT is untested. After long benchmark batches the page cache can hold 25 GiB of other models' files, and Flash-Next's start
refuses to load (resident complement against available RAM minus 8 GiB); evicting the files with `posix_fadvise(DONTNEED)` fixes it without root.

## Tuning sweeps (2026-11-04)

One run per row, warm decode, this box (12 logical CPUs), production stopped during the runs. Differences under about 0.5 tok/s are noise.

| Model | Setup | Decode tok/s |
|---|---|---|
| Bolt | base / DFlash depth 3 / + pinned / + 8 threads / + prefetch 8 | 17.6-18.3 / 21.4-24.0 / 22.3-23.2 / 23.8-25.5 / 25.9-26.1 |
| Super-VL 120B | prefetch 4 / 8 / 8 + 300 MiB reserve / 12 / 8 + 8 threads | 4.8-5.1 / 5.5-5.9 / 5.7-6.0 / 6.2-6.6 / 6.1-6.5 |
| Ornith 122B | base / q8 KV, no MTP block / + 8 threads | 3.2-3.5 / 4.1-4.2 / 7.8-8.4 |
| Ornith 122B, 120 tokens | 32k ctx / 16k / 8k (8 threads, q8 KV) | 8.5 / 9.1 / 8.9 |
| Ornith 122B | 16k with fixed slots (no rebalancing) | 7.9 (rebalancing is worth about +15%) |
| Ornith 122B | 16k with 6 / 10 / 12 threads | 8.6 / 6.1 / 3.8 (more than 8 threads hurts on 12 logical CPUs) |
| Ornith 122B | 16k, prefill chunk 256 | 9.6 (one run, not repeated) |

Ornith went from about 3.3 to about 9 tok/s: mostly 8 threads, then the packed q8 cache, no MTP block and a 16k context. Quality at 16k
was not checked. Ornith's 1024-token prefill chunk gave no gain.

## Work in progress: Bolt, Super-VL, Ornith (strata-nh and q36 paths)

Unfinished. None of these is a production model. Numbers are warm decode in tok/s on this card.

| Model | Setup | Decode | Notes |
|---|---|---|---|
| Nemotron 3.5 Bolt 74B-A9B UD-Q3_K_XL (IQ3) | 262k, 38 slots/layer, prefetch 4 | 13.60 | 29.86 GiB expert arena, 18.0% resident |
| Bolt 24GB QAT (NVFP4 stored as Q8_0) | 262k, 74 slots/layer, prefetch 4 | 15.33 | 15.88 GiB expert arena, 28.2% resident |
| Bolt 24GB QAT, real-prompt profile | 300 MiB reserve, 90 slots/layer | 16.10 | 42.4% resident; 18.4% above the IQ3 baseline, a mix of three changes |
| Bolt 24GB QAT with DFlash | draft depth 2 / 3 / 4 | 18.0-19.3 / 17.0-19.1 / 17.6-18.8 | the default depth ran at 6.3-6.5, worse than the no-DFlash 14.5-15.3; depth 3 is the keeper |
| Bolt 24GB QAT, native NVFP4 + q4_0 KV | 262k, prefetch 4 | 25.50 | from the research roofline table (section 12 of the Strata notes); not re-run in this fork |
| Super-VL 120B UD-IQ2_XXS | 62 slots, prefetch 4 / 61 slots, prefetch 8 | 4.12 / 5.10 | 41.29 GiB expert arena does not fit; 12% of experts resident |
| Ornith 1.5 122B UD-IQ3_XXS (Strata q36) | 5 slots/layer, 6 workers (earlier run; see the sweeps above for ~9 with 8 threads, q8 KV, 16k) | 6.4 | llama.cpp `-ncmoe 40` reaches 6.17. A separate 32-token bench (`ornith-bench-runs`) read 3.2-3.5; MTP depth 3 accepted 33% of drafts and ran at 2.6 |

Why Flash-Next decodes 2-5x faster than these: its experts are small (1.44-1.75 MB), 24-30% of them fit in VRAM and
the profile-seeded cache hits 78-95%, and MTP amortises the dense weights. Bolt is dense-heavy (6.4 GB read per token),
Ornith's experts are about twice as large and mostly on the CPU. The full analysis is in
`docs/research/2026-10-23-flash-next-vs-bolt-ornith-decode.md` and `2026-10-24-bolt-iq3-vs-24gb.md` in the llamaHub repo.

Where each stands: Bolt has the most measured progress (QAT artifact, profile, DFlash, native NVFP4 and quantized KV
work); Super-VL is measured but memory-bound; Ornith runs but is slower with MTP than without, so it is not finished.

## Not here

The llama.cpp work for this card lives in [`sterlenjohnson/llama.cpp`](https://github.com/sterlenjohnson/llama.cpp)
branch `cuda-ada-4060ti`. No Strata measurements are recorded there.
