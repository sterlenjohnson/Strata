# Ornith 1.5 122B IQ3_XXS

Measurements on the RTX 4060 Ti 16 GB test box (see [FORK.md](../../FORK.md) for the machine and the summary table). Warm decode in tok/s unless noted; the GPU power limit was 120 W for every number.

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

## Tuning sweeps

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

### Profile, threads, prefill chunk (2026-11-05)

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

### Earlier sweeps (2026-11-04)

| Model | Setup | Decode tok/s |
|---|---|---|
| Ornith 122B | base / q8 KV, no MTP block / + 8 threads | 3.2-3.5 / 4.1-4.2 / 7.8-8.4 |
| Ornith 122B, 120 tokens | 32k ctx / 16k / 8k (8 threads, q8 KV) | 8.5 / 9.1 / 8.9 |
| Ornith 122B | 16k with fixed slots (no rebalancing) | 7.9 (rebalancing is worth about +15%) |
| Ornith 122B | 16k with 6 / 10 / 12 threads | 8.6 / 6.1 / 3.8 (more than 8 threads hurts on 12 logical CPUs) |
| Ornith 122B | 16k, prefill chunk 256 | 9.6 (one run, not repeated) |

Ornith went from about 3.3 to about 9 tok/s: mostly 8 threads, then the packed q8 cache, no MTP block and a 16k context. Quality at 16k was not checked. The 1024-token prefill chunk gave no gain.

## Expert skipping on Ornith (2026-11-06)

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

**Does skipping hurt generated code?** Four coding prompts (binary search, word frequency, merge intervals, LRU cache; "reply with only the code block, with 3 asserts"), up to 1,400 tokens
at the default sampling of the engine's `--tokens` path (greedy), each extracted and run in Python: base 3 of 4 pass (the LRU answer's own asserts fail), w = 0.10 3 of 4 (the merge-intervals function is
correct but one of the model's asserts is wrong), w = 0.15 3 of 4 (the LRU answer is cut inside the reasoning block). Equal pass rates on four prompts is not a significant result, but there is no sign of a code regression
and decode was 1.9x faster at 0.10 and 3.5x at 0.15 on the long generations.

## Early work in progress

| Model | Setup | Decode | Notes |
|---|---|---|---|
| Ornith 1.5 122B UD-IQ3_XXS (Strata q36) | 5 slots/layer, 6 workers (earlier run; see the sweeps above for ~9 with 8 threads, q8 KV, 16k) | 6.4 | llama.cpp `-ncmoe 40` reaches 6.17. A separate 32-token bench (`ornith-bench-runs`) read 3.2-3.5; MTP depth 3 accepted 33% of drafts and ran at 2.6 |

## Performance over context (2026-11-06)

--pcie-frac 0.5, STRATA_Q36_SKIP_THR=0.10, 7 threads, q8 KV, chunk 2048, 131k context; 64 generated tokens per point, one run each, real repository text, each prompt prefilled from a fresh state.
A 4,096-token chunk did not fit at a 131k context (2,524 MiB needed, 2,331 free), so it was rerun at a 65k setting (below).

| Prompt tokens | Decode tok/s | Prefill tok/s |
|---|---|---|
| 2,048 | 13.8 | 619 |
| 8,192 | 11.8 | 632 |
| 16,384 | 10.8 | 632 |
| 32,768 | 9.5 | 592 |
| 65,536 | 6.5 | 395 |
| 131,000 | 4.3 | 177 |

Usable to about 32k (9.5 tok/s); decode halves again by 64k and prefill falls from about 620 to 177 tok/s at 131k.

![decode](../img/context-decode.svg)

![prefill](../img/context-prefill.svg)

### Chunk 4096 (65k context setting)

| Prompt tokens | Decode tok/s | Prefill tok/s |
|---|---|---|
| 2,048 | 13.1 | 629 |
| 8,192 | 11.5 | 1,094 |
| 16,384 | 10.2 | 993 |
| 32,768 | 8.7 | 797 |

Against chunk 2048 on the same run (14.4 / 11.9 / 10.9 / 9.4 decode, 605 / 632 / 631 / 592 prefill): a 4,096-token chunk is 1.7x faster to prefill at 8k and 1.35x at 32k, and costs about 3-7% of decode. At 2k tokens it is the same, because the prompt is a single chunk.
