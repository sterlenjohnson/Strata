# tandem.cpp and n-gram speculation on llama.cpp (2026-11-06)

llama.cpp f5222f25c with the tandem.cpp patch set (31 files), built for sm_89. Machine as in [FORK.md](../FORK.md). Greedy, `-c 16384 -np 1 -lm none`, 500 tokens, 2 runs per cell.
Models: Qwen3.6-35B-A3B UD-IQ3_XXS (fits the GPU) and UD-Q4_K_XL (about 21 GB, needs offload). Strata is not involved here.

Two tasks: **copy-edit** (1,776-token prompt: rewrite a Python file with one function renamed) and **prose** (original 400-word essay, 20-token prompt).

## N-gram speculation (no draft model, no extra VRAM)

| Setup | Copy-edit tok/s | Prose tok/s |
|---|---:|---:|
| IQ3_XXS on GPU, no speculation | 81 | 80-83 |
| `--spec-type ngram-mod` | **363-560** (all drafts accepted) | 82-101 |
| `--spec-type ngram-simple` | **397** (all drafts accepted) | 74-82 |

4.5-7x on copy-heavy work, nothing on original text. Use it for editing, refactoring and repeated-structure output.

## tandem.cpp (Q4_K_XL, experts offloaded)

| Setup | Copy-edit tok/s | Prose tok/s |
|---|---:|---:|
| `-ncmoe 30` (untuned) | 1.9 | 4.4-4.8 |
| `-ncmoe 24 -t 6` (tuned baseline) | 38-40 | 39 |
| `-ncmoe 28 --no-mmap` | failed to start | - |
| `-cmoe --moe-cache-mib auto` (tandem) | broken, see below | 57-64 |
| tandem + `ngram-mod` | 139 on one run, broken on the other | 62-75 |
| `-cmoe --moe-cache-mib 0` (no cache) | 1.6-2.2 | 2.3-2.9 |

Against the tuned baseline tandem is about +60% on prose (62 against 39). The 13x over `-ncmoe 30` is a baseline artifact.

## Correctness bug found

With the expert cache enabled, the 1,776-token copy-edit prompt produced `"\n\n"` and then EOS (2 tokens) in 4 of 5 runs, on both plain tandem and tandem + ngram-mod. With `--moe-cache-mib 0` the same prompt gives 500 normal tokens, so the fault is in the cached-expert path on a long prefill. Short prompts (20 tokens) were fine. Not reported upstream yet. Treat tandem output on long prompts as unverified until fixed; no quality (perplexity) check was run.

Not measured: Flash-Next and the other models on llama.cpp, long context, and the author's 97 tok/s claim (different GPU).
