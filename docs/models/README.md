# Models tested

One file per model. The summary table and the machine are in [FORK.md](../../FORK.md).

| Model | File |
|---|---|
| Flash-Next IQ3_XXS (production) | [flash-next.md](flash-next.md) |
| Qwen3.6-35B-A3B | [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) |
| Ornith 1.5 122B | [ornith-122b.md](ornith-122b.md) |
| Bolt 74B (QAT and Q3) | [bolt-74b.md](bolt-74b.md) |
| Super-VL 120B | [super-vl-120b.md](super-vl-120b.md) |
| Flint 40B (llama.cpp) | [flint-40b.md](flint-40b.md) |

Cross-model notes: [expert skipping](../expert-skipping.md), [KV codecs](../kv-codecs.md), [method and shared findings](../tuning-method.md).

## Why models of about 47 GB

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
