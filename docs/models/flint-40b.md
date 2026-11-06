# Flint 40B Q8_0 (llama.cpp)

Measurements on the RTX 4060 Ti 16 GB test box (see [FORK.md](../../FORK.md) for the machine and the summary table). Warm decode in tok/s unless noted; the GPU power limit was 120 W for every number.

Strata has no `flint` architecture. The prebuilt llama.cpp b11798 at `/mnt/AI/llcpp-11798/root` loads it (the box has CUDA 12.6 only, so the CUDA 13 runtime libraries were extracted from the NVIDIA pip wheels
into the scratchpad's `cu13/lib`). `llama-bench -ngl 99 -fa 1 -t 6 -ub 512 -p 512 -n 128 -r 2`, decode / prefill in tok/s: `-ncmoe` 40: 29.1 / 190; 38: 28.7 / 204; 36: 30.8 / 203; 35: 31.2 / 219; 34: 31.4 / 214; **33: 32.1 / 228**
(`-ncmoe 28` does not fit: 18.8 GiB of experts on the 16 GB card; 32 and below were not tried). At `-ncmoe 36`: 4 / 5 / 6 / 7 / 8 threads give 30.8 / 30.5 / 30.8 / 30.2 / 28.5. Larger micro-batches
help prefill a lot: pp2048 at `-ub` 1024 / 2048 reads 368 / 691 tok/s. `llama-server --n-cpu-moe 38 -c 8192` produced 200 tokens at about 29 tok/s. Use `-ncmoe 33 -t 6 -ub 2048`. Flint was only run for speed here
(the file is a "Preview" base model; the quick completion test continued the prompt as plain text).
