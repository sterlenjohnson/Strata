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
| `-cmoe --moe-cache-mib auto` (tandem) | ended early (see below) | 57-64 |
| tandem + `ngram-mod` | 139 on one run, ended early on the other | 62-75 |
| `-cmoe --moe-cache-mib 0` (no cache) | 1.6-2.2 | 2.3-2.9 |

Against the tuned baseline tandem is about +60% on prose (62 against 39). The 13x over `-ncmoe 30` is a baseline artifact.

## The "2-token stop" is not a tandem bug (follow-up sweep)

Earlier runs showed the cached-expert path ending the copy-edit prompt after `"\n\n"` + EOS. A follow-up sweep (prompts of 110, 401, 826, 1,268 tokens; 120 tokens each; raw `/completion`, no chat template) shows:

- The tuned baseline (`-ncmoe 24`) ends immediately (1 token, empty) on the 110, 401 and 826-token prompts too. The model is borderline between EOS and `<think>` on a raw prompt, so this is prompt behavior, not corruption.
- With the cache on, the first greedy token flips on some prompts: baseline always opens with `<think>`, tandem sometimes opens with "Here is the rewritten code" instead (2 of 8 cached runs). `-ub 64 -b 64` and `--moe-cache-mib 3000` did not make this go away or consistently produce it, so it is numeric drift in the split/cache path, consistent with the author's note that greedy output is not repeatable.
- Decode speed on these prompts: baseline 39-40 tok/s, tandem 45-60 tok/s (the larger figures at `-ub 64`).

So the earlier "broken" cells in the table mean "model chose EOS", not "wrong tokens". Redo copy-edit speed with the chat template before quoting a tandem copy-edit number. No perplexity check was run, so quality drift is still unmeasured.

## Flash-Next IQ3_XXS on tandem (llama.cpp)

Same build, `-ngl 99 -cmoe --moe-cache-mib auto -lm none -c 8192 -np 1`. Run under `systemd-run --user --scope -p MemoryMax=50G -p MemorySwapMax=0`: the first attempt without a cap rebooted the VM (47 GB model in a 60 GB VM), the capped one loaded and used about 45 GB. Greedy, 300 tokens, 2 runs per cell, raw prompts (the model echoed or continued the prompt, so outputs are not task answers).

| Setup | Copy-edit decode | Copy-edit prefill | Prose decode |
|---|---:|---:|---:|
| tandem | 23.0-25.4 | 168-214 | 29.9-31.5 |
| tandem + `ngram-mod` | 28.7-38.5 (drafts 81/81 and 234/286 accepted) | 165-219 | 28.6-33.1 |

Strata production on the same card does about 41 decode and about 1,200 prefill at short context, so tandem on llama.cpp is about 25-40% slower on decode and about 6x slower on prefill for this model. Flash-Next has no MTP or kv-resident path in llama.cpp here. N-gram helps the copy task (+25-60%) but far less than on the 35B, because the baseline is already limited by expert traffic. No long-context, quality or non-capped comparison was run.

Not measured: the other models on llama.cpp, long context, and the author's 97 tok/s claim (different GPU).
