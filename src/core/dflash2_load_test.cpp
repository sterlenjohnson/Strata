// src/core/dflash2_load_test.cpp - does a DFlash2 pack load on the GPU (GPU, no model).
//
//     dflash2_load_test <pack> [device]
//
// Prints the pack's hidden size and tap count as the drafter read them and the VRAM the load took. A pack whose shape the loader refuses
// prints the loader's reason and exits 1.
#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

using strata::core::Dflash2Drafter;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dflash2_load_test <pack> [device]\n"); return 2; }
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    size_t free0 = 0, total = 0;
    cudaSetDevice(device);
    cudaMemGetInfo(&free0, &total);
    Dflash2Drafter d;
    std::string err;
    if (!d.load(argv[1], device, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    std::printf("loaded %s: hidden %d, %d taps, %zu MiB of VRAM\n", argv[1], d.n_embd(), d.n_taps(), (free0 - free1) >> 20);
    return 0;
}
