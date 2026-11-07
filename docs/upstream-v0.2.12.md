# Moving to upstream v0.2.12 (2026-11-07)

Upstream is at v0.2.12 (engine 0.1.58), 303 commits past v0.2.10. The fork's engine trees (`/mnt/AI/Strata`, `/mnt/AI/Strata-ornith`) share no git history with it, so this is a port, not a merge. Branch `v0.2.12-4060ti` (tree `/mnt/AI/Strata-v0212`, pushed to `ghfork`) is upstream v0.2.12 plus the ports below.

| Part | State on v0.2.12 | Result |
|---|---|---|
| Flash-Next (`strata`) | upstream as is, production config | same speed as the fork build ([details](models/flash-next.md)) |
| Bolt / Super-VL (`strata-nh`) | ported: standalone target, src/nemotron_h, relu^2 experts, IQ1_S/Q2_K/Q3_K dot and dequant kernels, format-table entries | Bolt QAT pp 734, tg 29-33 against fork pp 731-734, tg 30-33 (same drafted tokens); Super-VL pp 282-292, tg 4.5-4.8 against pp 277-293, tg 4.5-4.8 |
| Ornith / 35B (`strata-q36`) | not ported: DFlash2 drafter, KV codecs, PCIe prefetch, expert skipping all conflict with upstream's `q36_engine.cu` (1,300 lines of changes) | stays on the fork build |

Why upstream has no Nemotron-H: v0.2.12 does not contain that engine, so `strata` does not switch to `strata-nh`; run `strata-nh` directly.
Bench settings: Bolt `--nvfp4 native --dflash <drafter> --dflash-n 3 --kv q8_0 --cpu-threads 8 --prefetch 12 --max-context 16384`, `STRATA_NH_SKIP_REL=0.5`; Super-VL `--cpu-threads 8 --prefetch 16`, no skipping. Both measured with the PCIe link at x4 (6.4 GB/s), so absolute speeds are below the x8 numbers elsewhere. Two runs per build.
New upstream opt-ins not tried: `STRATA_ADAPT_FETCH=1|2`, the DFlash2 CUDA drafter pack.
