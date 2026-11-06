# Method and shared findings

How the numbers were taken and what applies to every model.

Same box (6 cores / 12 threads in the VM, pinned; PCIe 4.0 x8, 13 GB/s measured; **the GPU power limit was 120 W** for every number in this file, against a 165 W default). Warm decode, one run each unless two numbers are
shown; differences under about 0.5 tok/s (Ornith, Super-VL) or 2 tok/s (Bolt) are noise.

One run per row, warm decode, this box (12 logical CPUs), production stopped during the runs. Differences under about 0.5 tok/s are noise.

**Prefill at larger prompts** (tok/s): Bolt 900 (2,048 and 4,096 tokens; 797 at 3,072), Super-VL 350 (290 at 3,072), Ornith 620 (470 at 3,072).

**Not measured, suggested:** the GPU power limit reads 120 W against a 165 W default, and the "SW Power Capping" counter shows about 892 s
of capping; raising it in LACT is untested. After long benchmark batches the page cache can hold 25 GiB of other models' files, and Flash-Next's start
refuses to load (resident complement against available RAM minus 8 GiB); evicting the files with `posix_fadvise(DONTNEED)` fixes it without root.

Why Flash-Next decodes 2-5x faster than these: its experts are small (1.44-1.75 MB), 24-30% of them fit in VRAM and
the profile-seeded cache hits 78-95%, and MTP amortises the dense weights. Bolt is dense-heavy (6.4 GB read per token),
Ornith's experts are about twice as large and mostly on the CPU. The full analysis is in
`docs/research/2026-10-23-flash-next-vs-bolt-ornith-decode.md` and `2026-10-24-bolt-iq3-vs-24gb.md` in the llamaHub repo.

Where each stands: Bolt has the most measured progress (QAT artifact, profile, DFlash, native NVFP4 and quantized KV
work); Super-VL is measured but memory-bound; Ornith runs but is slower with MTP than without, so it is not finished.
