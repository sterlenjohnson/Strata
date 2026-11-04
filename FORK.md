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
| Ornith 1.5 122B UD-IQ3_XXS (Strata q36) | 5 slots/layer, 6 workers | 6.4 | llama.cpp `-ncmoe 40` reaches 6.17. A separate 32-token bench (`ornith-bench-runs`) read 3.2-3.5; MTP depth 3 accepted 33% of drafts and ran at 2.6 |

Why Flash-Next decodes 2-5x faster than these: its experts are small (1.44-1.75 MB), 24-30% of them fit in VRAM and
the profile-seeded cache hits 78-95%, and MTP amortises the dense weights. Bolt is dense-heavy (6.4 GB read per token),
Ornith's experts are about twice as large and mostly on the CPU. The full analysis is in
`docs/research/2026-10-23-flash-next-vs-bolt-ornith-decode.md` and `2026-10-24-bolt-iq3-vs-24gb.md` in the llamaHub repo.

Where each stands: Bolt has the most measured progress (QAT artifact, profile, DFlash, native NVFP4 and quantized KV
work); Super-VL is measured but memory-bound; Ornith runs but is slower with MTP than without, so it is not finished.

## Not here

The llama.cpp work for this card lives in [`sterlenjohnson/llama.cpp`](https://github.com/sterlenjohnson/llama.cpp)
branch `cuda-ada-4060ti`. No Strata measurements are recorded there.
