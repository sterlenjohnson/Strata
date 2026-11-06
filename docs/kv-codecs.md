# KV-cache compression codecs (quality test)

Qwen3.6-35B-A3B engine. See also [that model's file](models/qwen3.6-35b-a3b.md).

`--kv-turbo k4v3 --kv-rot wht|planar|iso|ggml` (Qwen3.6-35B-A3B engine) rounds every K and V row through a codec as it is written and
writes the decoded values back. The cache stays fp32, so this measures quality, not memory; packed storage is not built.
Codecs: TurboQuant (norm, sign-flipped Walsh-Hadamard rotation over 128-value groups, Lloyd-Max centroids, norm corrected by the chosen
centroids' energy; format from TheTom/llama-cpp-turboquant), RotorQuant's PlanarQuant (random 2D rotation per pair) and IsoQuant (random
quaternion per 4-block), no norm correction as in its reference, and ggml's q8_0 / q4_0. The rotation parameters are seeded from this
engine's own random stream, not the reference's.

Perplexity of 1,500 teacher-forced tokens of technical prose after a 1,500-token prompt (UD-Q4_K_XL, one run each). fp32 cache: 10.512.
The ggml runs landed below it, so differences of about 0.04 are noise. Bytes are for K + V of one cell in one attention layer (512 values
each), calculated from block sizes (turbo4 66 B, turbo3 50 B, turbo2 about 34 B per 128 values; q8_0 34 B, q4_0 18 B per 32).

| K / V | Bytes | ggml | TurboQuant | PlanarQuant | IsoQuant |
|---|---|---|---|---|---|
| fp32 | 4096 | 10.512 | | | |
| 8 / 8 | 1088 | 10.491 | | | |
| 8 / 4 | 832 | 10.468 | | | |
| 4 / 4 | 528-576 | 10.505 | 10.515 | 10.781 | 10.798 |
| 4 / 3 | 464 | | 10.548 | 11.198 | 10.937 |
| 3 / 3 | 400 | | 10.658 | 11.251 | 10.920 |
| 3 / 2 | 336 | | 10.760 | 11.522 | 11.567 |
| 2 / 2 | about 272 | | 10.964 | 11.984 | 11.483 |

Reading: at 4 bits ggml q4_0 matches TurboQuant; TurboQuant's value is below 4 bits (K4/V3 +0.3% at 464 B, K3/V3 +1.4% at 400 B, K3/V2
+2.4%). On this model RotorQuant's rotations are worse than the Walsh-Hadamard one at every width. Norm correction changes little
(planar K3/V3 11.251 -> 11.121, iso 10.920 -> 10.925, TurboQuant K3/V2 10.760 -> 10.795 without it). One model, one text, approximate
tokenization; the rotations are applied per 128-value group (the head dim is 256), and the speed advantage claimed for RotorQuant is not
measured here.

With the drafters (200 tokens, depth 4 DFlash2 / depth 2 MTP): DFlash2 on the code prompt 90.3 tok/s uncompressed, 87.7 with K4/V3,
88.2 with K3/V3 (drafts accepted 73% / 69% / 74%); prose 77.8 / 74.2 / 68.7 (54% / 52% / 57%). MTP code 71.8 -> 76.5 (83% -> 86%), prose
84.4 -> 80.9 (85% -> 77%). All within the run-to-run noise: compressed KV does not break drafting. The DFlash2 drafter's own ring is
separate and untouched.
