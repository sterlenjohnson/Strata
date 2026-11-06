# Expert skipping: drop low-weight experts that would run on the CPU (2026-11-06)

Per-model results: [Ornith](models/ornith-122b.md), [Bolt](models/bolt-74b.md), [Super-VL](models/super-vl-120b.md), [Flash-Next](models/flash-next.md).

Experts with no VRAM slot cost a CPU pass (or a PCIe copy) each, and many of them carry a small routing weight. Skipping them removes that work. Only
CPU-side entries are ever skipped; experts resident in VRAM or already prefetched are always computed. Off by default and bit-exact when unset.

| Engine | Env var | Rule |
|---|---|---|
| Ornith (`strata-q36`) | `STRATA_Q36_SKIP_THR=w` | skip a slot-less expert whose routing weight is below w (weights sum to 1 over the top 8) |
| Super-VL, Bolt (`strata-nh`) | `STRATA_NH_SKIP_REL=f` (or `STRATA_NH_SKIP_THR=w`) | skip below f times the row's largest weight (absolute weights depend on top-k and the model's scale, so use REL) |
| Flash-Next (`strata`) | `STRATA_SKIP_THR=w` | same rule as Ornith; `STRATA_SKIP_HIST=1` prints the weight quantiles; works beside the upstream rank-based `STRATA_ROUTE_TAIL_SKIP` |

Also available: `STRATA_Q36_SKIP_RENORM=1` / `STRATA_NH_SKIP_RENORM=1` (rescale the kept weights; worse, see below), `STRATA_Q36_NLL=1` / `STRATA_NH_NLL=1` with `--tokens PROMPT --force IDS`
(teacher-forced perplexity through the skipping path; the output line `expert skipping (...): N of M CPU-side entries skipped` gives the skip rate).

Repeat runs at the end of the session (same settings): Ornith base 12.4, `STRATA_Q36_SKIP_THR` 0.10 gives 16.9 and 0.12 gives 22.1 tok/s; Super-VL base 6.5 / 7.0, `STRATA_NH_SKIP_REL=0.5` 7.8 / 8.1 in two further runs (+17%).

**How to pick a value.** Start with the table, run `STRATA_Q36_NLL=1` on a text like your workload (`--tokens` prompt, `--force` continuation), and take the largest threshold whose perplexity
stays within about 3% of w = 0. Check the skip-rate line: a rate above 90% means the model is running almost entirely on the VRAM-resident experts.

**Files changed** (all uncommitted in the working trees):
- `/mnt/AI/Strata-ornith/src/qwen35/q36_engine.cu`: `STRATA_Q36_SKIP_THR`, `SKIP_RENORM`, mass cap, `DEFER`, `NLL`, `PROF`, `PREDICT`.
- `/mnt/AI/Strata-ornith/src/nemotron_h/nh_engine.cu`: `STRATA_NH_SKIP_REL/THR/RENORM`, `STRATA_NH_NLL` (skip decided before the PCIe share, so skipped experts are never uploaded).
- `/mnt/AI/Strata-main`: `src/core/expert_source.cpp` and `include/strata/core/expert_source.hpp` (decision and counters), `src/core/verify.cpp` and `include/strata/core/verify.hpp`
  (weights to the host, renormalised combine), `src/kernels/cuda/elementwise.cu` and `include/strata/kernels/elementwise.hpp` (doorbell publishes weights), `src/program/generate.cpp` (stats line).
