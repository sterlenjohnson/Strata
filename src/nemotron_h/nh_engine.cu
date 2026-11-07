// src/nemotron_h/nh_engine.cu - strata-nh: the engine path for NVIDIA Nemotron-H MoE (GGUF architecture
// nemotron_h_moe: Nemotron 3.5 Super VL 120B-A12B, unsloth's GGUFs).  A driver of its own beside Flash-Next's and
// Qwen3.6's (their paths stay untouched), built from Strata's kernels plus its own (nh_kernels.cuh):
//
//   * 88 flat sublayers, each x += op(rms(x) * attn_norm): 40 Mamba-2 mixers, 8 attention mixers, 40 MoE blocks
//     (the GGUF's per-layer head_count_kv / feed_forward_length say which), then output_norm and the head;
//   * Mamba-2: in_proj (z | xBC | dt), the causal conv + bias + SiLU, the selective scan (128 heads x 64, state 128,
//     8 groups), y + D x, the SiLU(z) gate and the grouped RMS norm (1024 per group), out_proj;
//   * attention: 32 query heads over 2 KV heads, head 128, NO positional encoding (NoPE), an FP16 KV cache;
//   * MoE: the sigmoid router with the selection bias (512 experts, top 22, normalized, x 5.0), the latent
//     projections 4096 -> 1024 -> 4096 around the routed experts down(relu(up x)^2), the shared expert
//     down(relu(up h)^2) on the full width;
//   * the routed experts are 78 GB: the hybrid mode always - per MoE layer the same number of VRAM slots, a batch of
//     up to --hybrid-rows rows runs its resident experts on the GPU and the others on the CPU pool (ggml-cpu, read
//     straight from the mapped GGUF), a bigger prompt chunk streams every non-resident expert through a ring of
//     VRAM buffers (a copy thread feeding a copy stream) into per-expert FP16 GEMMs;
//   * the GGUF's MTP block (blk.88: eh_proj over [enorm(e) | hnorm(h)], attention, MoE, the LayerNorm head norm)
//     drafts for greedy requests (--mtp N), verified by the main model (the Mamba state is committed afterwards);
//   * images (GENI): strata-vision's rows (llama.cpp mtmd, the mmproj's RADIO tower + projector) in place of the
//     prompt's <image> tokens (the server writes <img> <image> x n </img>).
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / GENI / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/nemotron_h's parity interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/nemotron_h/cpu_experts.hpp"
#include "strata/nemotron_h/engine.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/nemotron_h/slot_plan.hpp"

#include "nh_kernels.cuh"
#include "ggml.h"   // nvfp4_check: ggml-cpu's NVFP4 to_float
#include "nh_dflash.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

namespace K = strata::kernels;
using namespace nh;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-nh: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }

constexpr int kImagePad = 18;          // <image>: the rows strata-vision's embeddings take (between <img> and </img>)
constexpr int kEos = 11;               // <|im_end|> (tokenizer.ggml.eos_token_id)
constexpr int kEos2 = 2;               // </s>
constexpr size_t kTail = 4096;         // zeroed bytes after the last slot (a kernel reading a little past a matrix)

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
    float s2 = 1.f;               // native NVFP4 (type 40, --nvfp4 native): the per-tensor .scale, on the output
};

// --nvfp4 native: NVFP4 matrices stay NVFP4 in VRAM (4.5 bits a weight; decode on iq_kernels' NVFP4 MMVQ, a prompt
// chunk on its FP16 dequant + the FP16 GEMM, the .scale on the FP32 output as llama.cpp's build_lora_mm does).
// Off (default): the Q8_0 transcode below.
bool g_nvfp4_native = false;

enum class Kind { Mamba, Attn, Moe };

struct Moe {
    const float *router = nullptr, *bias = nullptr;   // [E][N] f32, [E]
    Mat lat_down, lat_up, sh_up, sh_down;
    K::NativeExpertLayout L;                           // [up | down] blobs (ungated)
    size_t stride = 0;                                 // a slot's bytes (16-aligned)
    const uint8_t *src_up = nullptr, *src_down = nullptr;   // the mapped GGUF's expert tensors
    size_t up_bytes = 0, d_bytes = 0;                  // per expert
    uint8_t* host = nullptr;                           // the pinned arena's copy: blob e at host + e * stride (null: none)
    /// expert e's up / down rows in host memory: the arena's blob, else the mapped file's slices
    const uint8_t* up_of(int e) const { return host ? host + (size_t) e * stride : src_up + (size_t) e * up_bytes; }
    const uint8_t* down_of(int e) const { return host ? host + (size_t) e * stride + L.down_off : src_down + (size_t) e * d_bytes; }
    strata::nemotron_h::CpuFmt fmt;
    uint8_t* slots = nullptr;                          // n_slots blobs, `stride` apart
    int n_slots = 0;
    // per-layer geometry (Bolt varies FF/E across MoE layers; Super-VL is uniform)
    int layer_ff = 0;      // expert FF width for this layer
    int layer_e = 0;       // expert count for this layer
};

struct Layer {
    Kind kind = Kind::Mamba;
    int idx = 0;                     // its index among the layers of its kind (state / KV / slot tables)
    const float* norm = nullptr;     // attn_norm (every sublayer's pre-norm)
    // Mamba-2
    Mat ssm_in, ssm_out;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    // MoE
    Moe moe;
};

struct Geo {
    int n_layers = 0, n_all = 0, N = 0, V = 0;
    int H = 0, HKV = 0;
    int E = 0, top_k = 0, FF = 0, FS = 0, LAT = 0;
    bool w_norm = true;
    float w_scale = 1.f;
    int ssm_conv = 4, ssm_N = 128, ssm_G = 8, ssm_inner = 8192, ssm_H = 128;
    int conv_ch = 0, in_proj = 0;
    float eps = 1e-5f;
};

struct Model {
    Geo g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are read from it
    std::vector<Layer> layers;
    int n_mamba = 0, n_attn = 0, n_moe = 0;
    // the MTP block (blk.<n_layers>): eh_proj over [enorm(e) | hnorm(h)], attention + MoE, shared_head_norm
    bool has_mtp = false;
    Layer mtp_attn, mtp_moe;          // its attention (KV index n_attn) and its MoE (slot index n_moe)
    const float* mtp_post_norm = nullptr;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    size_t dense_bytes = 0;
};

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) u = sign;
        else {
            uint32_t m = mantissa, shift = 0;
            while ((m & 0x400) == 0) { m <<= 1; ++shift; }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) u = sign | 0x7f800000u | (mantissa << 13);
    else u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else {
        std::fprintf(stderr, "strata-nh: %s is %s, expected F32, BF16 or F16\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

const strata::MetaValue* meta(const strata::GgufModel& gm, const std::string& key) { return gm.meta().get(key); }

// ---- NVFP4 -> Q8_0 at load (the native MMVQ has no NVFP4 path).  Bolt's 24GB QAT file stores the Mamba in/out
// projections as NVFP4 (64-value blocks: four UE4M3 scales of 16 values, E2M1 codes) with a per-tensor F32 `.scale`
// that llama.cpp multiplies into the matmul's output (build_lora_mm's w_s).  Each 32-value Q8_0 block takes the exact
// dequantized values (times that scale) rounded to int8 against the block's max: the usual Q8_0 error, ~2x the
// bytes of NVFP4 (8.5 vs 4.5 bits a weight).
constexpr int kTypeNvfp4 = 40, kTypeQ8_0 = 8;
float ue4m3_to_f32(uint8_t x) {   // ggml_ue4m3_to_fp32, including its 0.5
    if (x == 0 || x == 0x7F) return 0.f;
    const int e = (x >> 3) & 0xF, mt = x & 0x7;
    const float raw = e == 0 ? std::ldexp((float) mt, -9) : std::ldexp(1.f + (float) mt / 8.f, e - 7);
    return raw * 0.5f;
}
std::vector<uint8_t> nvfp4_to_q8_0(const uint8_t* src, int64_t n_in, int64_t n_out, float scale2) {
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   // kvalues_mxfp4
    const int64_t nb4 = n_in / 64, nb8 = n_in / 32;
    std::vector<uint8_t> out((size_t) (n_out * nb8 * 34));
    auto rows = [&](int64_t r0, int64_t r1) {
        float v[64];
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t* rb = src + (size_t) (r * nb4 * 36);
            uint8_t* ob = out.data() + (size_t) (r * nb8 * 34);
            for (int64_t b = 0; b < nb4; ++b) {
                const uint8_t* blk = rb + b * 36;   // d[4], qs[32]
                for (int s = 0; s < 4; ++s) {
                    const float d = ue4m3_to_f32(blk[s]) * scale2;
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t q = blk[4 + s * 8 + j];
                        v[s * 16 + j] = kv[q & 0x0F] * d;
                        v[s * 16 + j + 8] = kv[q >> 4] * d;
                    }
                }
                for (int h = 0; h < 2; ++h) {
                    const float* x = v + 32 * h;
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
                    const float dq = amax / 127.f, id = dq != 0.f ? 1.f / dq : 0.f;
                    uint8_t* q8 = ob + (size_t) ((2 * b + h) * 34);
                    const __half hd = __float2half(dq);
                    std::memcpy(q8, &hd, 2);
                    for (int i = 0; i < 32; ++i) q8[2 + i] = (uint8_t) (int8_t) std::lrintf(x[i] * id);
                }
            }
        }
    };
    const int nt = (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) ts.emplace_back(rows, n_out * t / nt, n_out * (t + 1) / nt);
    for (auto& t : ts) t.join();
    return out;
}

// STRATA_NH_NVFP4_CHECK=1 (--nvfp4 native): each NVFP4 matrix checked at load on 256 sampled rows against ggml-cpu's
// own to_float (dequantize_row_nvfp4): (1) iq_dequant_f32 bitwise, (2) iq_dequant_f16 (the prompt's) bitwise against
// FP16 of it, (3) the decode MMVQ (+ .scale) against a double dot of ggml's weights with the dequantized q8_1
// activation, beside the same for the Q8_0 transcode (what --nvfp4 q8 runs).  Any (1)/(2) mismatch or a (3) rel-RMS
// above 1e-5 exits.
void nvfp4_check(const std::string& name, const uint8_t* src, const uint8_t* d, int n_in, int n_out, float scale2) {
    static cudaStream_t cs = [] { cudaStream_t c; CK(cudaStreamCreate(&c)); return c; }();
    const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    const size_t rb = (size_t) n_in / 64 * 36;
    if (tr->blck_size != 64 || tr->type_size != 36) { std::fprintf(stderr, "nvfp4 check: ggml's block is not 64/36\n"); std::exit(1); }
    const int nr = std::min(256, n_out);
    std::vector<int> rows((size_t) nr);
    for (int i = 0; i < nr; ++i) rows[(size_t) i] = (int) ((int64_t) i * n_out / nr);
    std::vector<float> ref((size_t) n_in), g32((size_t) n_in);
    std::vector<__half> g16((size_t) n_in);
    float* d32 = nullptr;
    __half* d16 = nullptr;
    CK(cudaMalloc(&d32, (size_t) n_in * 4));
    CK(cudaMalloc(&d16, (size_t) n_in * 2));
    // the activation: a fixed random row, its q8_1 image (the engine's quantizer) and that image dequantized
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n_in), xd((size_t) n_in);
    for (auto& v : x) v = nd(rng);
    const size_t qb = K::native_q8_1_bytes(n_in, 1);
    std::vector<uint8_t> hq(qb);
    float *dx = nullptr, *dy = nullptr, *dy8 = nullptr;
    void* dq = nullptr;
    CK(cudaMalloc(&dx, (size_t) n_in * 4));
    CK(cudaMalloc(&dq, qb));
    CK(cudaMalloc(&dy, (size_t) n_out * 4));
    CK(cudaMalloc(&dy8, (size_t) n_out * 4));
    CK(cudaMemcpy(dx, x.data(), (size_t) n_in * 4, cudaMemcpyHostToDevice));
    K::native_quantize_q8_1(dx, dq, n_in, 1, cs);
    CK(cudaMemcpy(hq.data(), dq, qb, cudaMemcpyDeviceToHost));
    for (int b = 0; b < n_in / 32; ++b) {
        const uint8_t* blk = hq.data() + (size_t) b * 36;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const float db = f16_to_f32(dh);
        for (int i = 0; i < 32; ++i) xd[(size_t) b * 32 + i] = (float) (int8_t) blk[4 + i] * db;
    }
    K::iq_mmvq(kTypeNvfp4, d, dq, dy, n_in, n_out, 1, cs);
    // the transcode's result on the same activation (scale folded into the Q8_0 rows)
    const std::vector<uint8_t> q8 = nvfp4_to_q8_0(src, n_in, n_out, scale2);
    uint8_t* dw8 = nullptr;
    CK(cudaMalloc(&dw8, q8.size() + kTail));
    CK(cudaMemcpy(dw8, q8.data(), q8.size(), cudaMemcpyHostToDevice));
    K::native_mmvq(kTypeQ8_0, dw8, dq, dy8, n_in, n_out, 1, cs);
    std::vector<float> y((size_t) n_out), y8((size_t) n_out);
    CK(cudaStreamSynchronize(cs));
    CK(cudaMemcpy(y.data(), dy, (size_t) n_out * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y8.data(), dy8, (size_t) n_out * 4, cudaMemcpyDeviceToHost));
    long long bad32 = 0, bad16 = 0;
    double e2 = 0, e2q8 = 0, r2 = 0;
    for (int r : rows) {
        tr->to_float(src + (size_t) r * rb, ref.data(), n_in);
        K::iq_dequant_f32(kTypeNvfp4, d + (size_t) r * rb, n_in, d32, cs);
        K::iq_dequant_f16(kTypeNvfp4, d + (size_t) r * rb, n_in, (uint16_t*) d16, cs);
        CK(cudaStreamSynchronize(cs));
        CK(cudaMemcpy(g32.data(), d32, (size_t) n_in * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g16.data(), d16, (size_t) n_in * 2, cudaMemcpyDeviceToHost));
        double dot = 0;
        for (int i = 0; i < n_in; ++i) {
            if (std::memcmp(&ref[(size_t) i], &g32[(size_t) i], 4)) ++bad32;
            const __half h = __float2half(ref[(size_t) i]);
            if (std::memcmp(&h, &g16[(size_t) i], 2)) ++bad16;
            dot += (double) ref[(size_t) i] * xd[(size_t) i];
        }
        dot *= scale2;
        const double yg = (double) y[(size_t) r] * scale2;   // the engine scales the MMVQ's output by s2
        e2 += (yg - dot) * (yg - dot);
        e2q8 += ((double) y8[(size_t) r] - dot) * ((double) y8[(size_t) r] - dot);
        r2 += dot * dot;
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-300)), relq8 = std::sqrt(e2q8 / std::max(r2, 1e-300));
    std::fprintf(stderr, "strata-nh: nvfp4 check %s [%d x %d, scale %g]: f32 dequant %lld / f16 dequant %lld of %lld "
                 "values differ from ggml to_float; MMVQ rel-RMS %.2e (the Q8_0 transcode: %.2e)\n", name.c_str(), n_out,
                 n_in, scale2, bad32, bad16, (long long) nr * n_in, rel, relq8);
    CK(cudaFree(d32)); CK(cudaFree(d16)); CK(cudaFree(dx)); CK(cudaFree(dq)); CK(cudaFree(dy)); CK(cudaFree(dy8)); CK(cudaFree(dw8));
    if (bad32 || bad16 || !(rel < 1e-5)) { std::fprintf(stderr, "strata-nh: nvfp4 check FAILED on %s\n", name.c_str()); std::exit(1); }
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::MetaValue* arch = meta(gm, "general.architecture");
    if (!arch || arch->s != "nemotron_h_moe") {
        err = "strata-nh runs Nemotron-H MoE (nemotron_h_moe) files only";
        return false;
    }
    const std::string p = "nemotron_h_moe.";
    auto num = [&](const std::string& k, double dflt) {
        const strata::MetaValue* v = meta(gm, p + k);
        return v ? v->num() : dflt;
    };
    // A key that is a scalar (Super-VL) or a per-sublayer array (Bolt: expert_used_count, expert_feed_forward_length
    // are arrays of 88 with 0 on the non-MoE sublayers).  Returns the one non-zero value; `mixed` is set when the
    // array's non-zero entries differ.  The reader keeps only an array's first 64 items, so the per-layer widths are
    // taken from the tensors below, never from these arrays.
    auto uniform_num = [&](const std::string& k, double dflt, bool& mixed) -> int {
        mixed = false;
        const strata::MetaValue* v = meta(gm, p + k);
        if (!v) return (int) dflt;
        if (v->type != strata::MetaType::ARRAY) return (int) v->num();
        int val = 0;
        for (const auto& it : v->items) {
            if (!it.is_num() || it.num() == 0) continue;
            if (val == 0) val = (int) it.num();
            else if ((int) it.num() != val) mixed = true;
        }
        return val ? val : (int) dflt;
    };
    Geo& g = m.g;
    g.n_all = (int) num("block_count", 0);
    const int nextn = (int) num("nextn_predict_layers", 0);
    g.n_layers = g.n_all - nextn;
    g.N = (int) num("embedding_length", 0);
    g.V = (int) num("vocab_size", 0);
    g.H = (int) num("attention.head_count", 0);
    g.E = (int) num("expert_count", 0);
    bool mixed_k = false, mixed_ff = false, mixed_fs = false;
    g.top_k = uniform_num("expert_used_count", 0, mixed_k);
    g.FF = uniform_num("expert_feed_forward_length", 0, mixed_ff);   // a first guess: the max over the tensors below
    g.FS = uniform_num("expert_shared_feed_forward_length", 0, mixed_fs);
    if (mixed_k || mixed_fs) {
        // the router's top-k and the shared expert's width are one value in every buffer and kernel of this driver
        err = mixed_k ? "expert_used_count differs between MoE layers, which this driver does not take (one top-k)"
                      : "expert_shared_feed_forward_length differs between MoE layers, which this driver does not take";
        return false;
    }
    g.LAT = (int) num("moe_latent_size", 0);
    g.w_norm = num("expert_weights_norm", 1) != 0;
    g.w_scale = (float) num("expert_weights_scale", 1.0);
    g.ssm_conv = (int) num("ssm.conv_kernel", 4);
    g.ssm_N = (int) num("ssm.state_size", 128);
    g.ssm_G = (int) num("ssm.group_count", 8);
    g.ssm_inner = (int) num("ssm.inner_size", 8192);
    g.ssm_H = (int) num("ssm.time_step_rank", 128);
    g.eps = (float) num("attention.layer_norm_epsilon", 1e-5);
    if (const strata::MetaValue* v = meta(gm, p + "attention.layer_norm_rms_epsilon")) g.eps = (float) v->num();
    g.conv_ch = g.ssm_inner + 2 * g.ssm_G * g.ssm_N;
    g.in_proj = 2 * g.ssm_inner + 2 * g.ssm_G * g.ssm_N + g.ssm_H;
    const int key_len = (int) num("attention.key_length", 128);
    if (g.V == 0) {
        if (const strata::TensorInfo* t = gm.find("token_embd.weight")) g.V = (int) t->shape[1];
    }
    if (g.n_layers <= 0 || g.N % 256 || key_len != kHD || g.ssm_conv != 4 || g.ssm_N != kSsmN ||
        g.ssm_inner / g.ssm_H != kSsmP || g.ssm_H % g.ssm_G || g.E > 512 || g.top_k > kMaxTopK || g.top_k < 1 ||
        g.LAT % 256 || g.FF % 32 || g.FS % 256 || g.ssm_inner % g.ssm_G) {
        err = "a geometry this driver's kernels do not take (head 128, conv 4, state 128, head_dim 64, <= 512 experts, "
              "top-k <= 32, widths in whole blocks)";
        return false;
    }
    // a sublayer's kind from its tensors (the per-layer arrays head_count_kv / feed_forward_length say the same, but the
    // reader keeps only an array's first 64 entries: there are 88 sublayers)
    auto has = [&](int64_t l, const char* s) { return gm.find("blk." + std::to_string(l) + "." + s) != nullptr; };
    uint64_t bytes = 0, nvfp4_bytes = 0;
    int nvfp4_mats = 0;
    auto find =[&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-nh: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        // a per-tensor `.scale` (NVFP4's second-level scale) is folded into the weights; on any other type it would
        // be silently dropped, so it is refused
        const std::string base = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
                                     ? name.substr(0, name.size() - 7) : name;
        size_t ss = 0;
        const strata::TensorInfo* ts = gm.find(base + ".scale", &ss);
        float scale2 = 1.f;
        if (ts) {
            if (ts->elements() != 1 || ts->type != 0) {
                std::fprintf(stderr, "strata-nh: %s.scale is not one F32 value\n", base.c_str());
                std::exit(1);
            }
            std::memcpy(&scale2, gm.shard(ss).tensor_data(*ts), 4);
            if (w.type != kTypeNvfp4) {
                std::fprintf(stderr, "strata-nh: %s has a .scale on a %s matrix, which this driver does not apply\n",
                             name.c_str(), t->type_name());
                std::exit(1);
            }
        }
        if (w.type == kTypeNvfp4 && g_nvfp4_native) {
            if (w.n_in % 256) {   // iq_dequant_f16 takes whole 256-value superblocks
                std::fprintf(stderr, "strata-nh: %s is NVFP4 with a row of %d (not whole 256s)\n", name.c_str(), w.n_in);
                std::exit(1);
            }
            const uint64_t nb = strata::tensor_payload_bytes(*t);
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, nb + kTail));
            CK(cudaMemset(d + nb, 0, kTail));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            if (std::getenv("STRATA_NH_NVFP4_CHECK")) nvfp4_check(name, gm.shard(s).tensor_data(*t), d, w.n_in, w.n_out, scale2);
            bytes += nb;
            ++nvfp4_mats;
            nvfp4_bytes += nb;
            w.d = d;
            w.s2 = scale2;
            return w;
        }
        if (w.type == kTypeNvfp4) {
            if (w.n_in % 64) {
                std::fprintf(stderr, "strata-nh: %s is NVFP4 with a row of %d (not whole 64-blocks)\n", name.c_str(), w.n_in);
                std::exit(1);
            }
            const std::vector<uint8_t> q = nvfp4_to_q8_0(gm.shard(s).tensor_data(*t), w.n_in, w.n_out, scale2);
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, q.size()));
            CK(cudaMemcpy(d, q.data(), q.size(), cudaMemcpyHostToDevice));
            bytes += q.size();
            ++nvfp4_mats;
            nvfp4_bytes += q.size();
            w.type = kTypeQ8_0;
            w.d = d;
            return w;
        }
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-nh: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    auto load_moe = [&](int64_t l, Moe& y) -> bool {
        y.router = f32(L(l, "ffn_gate_inp.weight"));   // shapes checked against the experts' below
        y.bias = f32(L(l, "exp_probs_b.bias"));
        y.lat_down = mat(L(l, "ffn_latent_down.weight"));
        y.lat_up = mat(L(l, "ffn_latent_up.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        size_t su = 0, sd = 0;
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        // this layer's own geometry, from its tensors (Bolt: 240-496 experts and FF 1536-2560 by layer; Super-VL
        // is uniform).  up [LAT][FF][E], down [FF][LAT][E], router [N][E], bias [E].
        y.layer_ff = (int) tu->shape[1];
        y.layer_e = (int) tu->shape[2];
        size_t sr = 0, sb = 0;
        const strata::TensorInfo* tr = find(L(l, "ffn_gate_inp.weight"), sr);
        const strata::TensorInfo* tb = find(L(l, "exp_probs_b.bias"), sb);
        char why[256];
        std::snprintf(why, sizeof why, "up %lldx%lldx%lld, down %lldx%lldx%lld, router %lldx%lld, bias %lld; latent %d",
                      (long long) tu->shape[0], (long long) tu->shape[1], (long long) tu->shape[2],
                      (long long) td->shape[0], (long long) td->shape[1], (long long) td->shape[2],
                      (long long) tr->shape[0], (long long) tr->shape[1], (long long) tb->shape[0], g.LAT);
        if ((int) tu->shape[0] != g.LAT || (int) td->shape[1] != g.LAT || (int) td->shape[0] != y.layer_ff ||
            (int) td->shape[2] != y.layer_e || (int) tr->shape[0] != g.N || (int) tr->shape[1] != y.layer_e ||
            (int) tb->shape[0] != y.layer_e) {
            err = L(l, "expert tensors have inconsistent shapes: ") + why;
            return false;
        }
        if (y.layer_e < g.top_k || y.layer_e > 512 || y.layer_ff <= 0 || y.layer_ff % 32) {
            // k_route takes <= 512 experts (one block); the block sizes are checked by the expert kernels below
            err = L(l, "expert geometry this driver does not take (<= 512 experts, >= top-k, FF a multiple of 32): ") + why;
            return false;
        }
        if (!K::native_expert_ungated_supported((int) tu->type, (int) td->type, g.LAT, y.layer_ff)) {
            err = L(l, "experts are ") + tu->type_name() + "/" + td->type_name() +
                  ", which the native expert kernels do not take";
            return false;
        }
        y.L = K::native_expert_layout_ungated((int) tu->type, (int) td->type, g.LAT, y.layer_ff);
        y.stride = (y.L.bytes + 255) / 256 * 256;
        y.up_bytes = (size_t) y.layer_ff * y.L.gu_row;
        y.d_bytes = (size_t) g.LAT * y.L.d_row;
        y.src_up = gm.shard(su).tensor_data(*tu);
        y.src_down = gm.shard(sd).tensor_data(*td);
        if (!strata::nemotron_h::cpu_fmt((int) tu->type, (int) td->type, g.LAT, y.layer_ff, y.fmt, err)) return false;
        if (y.fmt.up_row != y.L.gu_row || y.fmt.d_row != y.L.d_row) {
            err = "the CPU and GPU expert row sizes differ";
            return false;
        }
        return true;
    };
    m.layers.resize((size_t) g.n_layers);
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.norm = f32(L(l, "attn_norm.weight"));
        if (has(l, "attn_q.weight")) {
            y.kind = Kind::Attn;
            y.idx = m.n_attn++;
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            const int hkv = y.wk.n_out / kHD;
            if (g.HKV == 0) g.HKV = hkv;
            if (hkv != g.HKV || y.wq.n_out != g.H * kHD || y.wv.n_out != hkv * kHD) {
                err = L(l, "attention has an unexpected shape");
                return false;
            }
        } else if (has(l, "ffn_up_exps.weight")) {
            y.kind = Kind::Moe;
            y.idx = m.n_moe++;
            if (!load_moe(l, y.moe)) return false;
        } else {
            y.kind = Kind::Mamba;
            y.idx = m.n_mamba++;
            y.ssm_in = mat(L(l, "ssm_in.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.conv_w = f32(L(l, "ssm_conv1d.weight"));
            size_t s = 0;
            if (gm.find(L(l, "ssm_conv1d.bias"), &s)) y.conv_b = f32(L(l, "ssm_conv1d.bias"));
            y.dt_b = f32(L(l, "ssm_dt.bias"));
            y.A = f32(L(l, "ssm_a"));
            y.D = f32(L(l, "ssm_d"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
            if (y.ssm_in.n_out != g.in_proj) { err = L(l, "ssm_in has an unexpected width"); return false; }
        }
    }
    if (m.n_attn == 0 || g.H % std::max(1, g.HKV)) { err = "no attention layer, or heads not a multiple of KV heads"; return false; }
    m.has_mtp = nextn > 0;
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
        m.mtp_attn.kind = Kind::Attn;
        m.mtp_attn.idx = m.n_attn;
        m.mtp_attn.norm = f32(L(l, "attn_norm.weight"));
        m.mtp_attn.wq = mat(L(l, "attn_q.weight"));
        m.mtp_attn.wk = mat(L(l, "attn_k.weight"));
        m.mtp_attn.wv = mat(L(l, "attn_v.weight"));
        m.mtp_attn.wo = mat(L(l, "attn_output.weight"));
        m.mtp_post_norm = f32(L(l, "post_attention_norm.weight"));
        m.mtp_moe.kind = Kind::Moe;
        m.mtp_moe.idx = m.n_moe;
        m.mtp_moe.norm = m.mtp_post_norm;
        if (!load_moe(l, m.mtp_moe.moe)) return false;
    }
    {
        // every buffer sized by g.E / g.FF (router logits, histograms, the slot tables, the expert scratch, the
        // prompt GEMM's up rows) takes the widest layer; each layer computes with its own layer_e / layer_ff
        int e_max = 0, e_min = 1 << 30, ff_max = 0, ff_min = 1 << 30;
        auto take = [&](const Moe& y) {
            e_max = std::max(e_max, y.layer_e); e_min = std::min(e_min, y.layer_e);
            ff_max = std::max(ff_max, y.layer_ff); ff_min = std::min(ff_min, y.layer_ff);
        };
        for (const Layer& y : m.layers)
            if (y.kind == Kind::Moe) take(y.moe);
        if (m.has_mtp) take(m.mtp_moe.moe);
        if (e_max == 0) { err = "no MoE layer"; return false; }
        if (e_max > g.E && g.E != 0) std::fprintf(stderr, "strata-nh: expert_count %d < a layer's %d experts: using %d\n", g.E, e_max, e_max);
        g.E = std::max(g.E, e_max);
        g.FF = ff_max;
        if (e_min != e_max || ff_min != ff_max || mixed_ff)
            std::fprintf(stderr, "strata-nh: per-layer experts: %d-%d experts, FF %d-%d (buffers sized for %d / %d)\n",
                         e_min, e_max, ff_min, ff_max, g.E, g.FF);
        if (nvfp4_mats)
            std::fprintf(stderr, g_nvfp4_native ? "strata-nh: %d NVFP4 matrices kept native (%.2f GiB in VRAM, --nvfp4 native)\n"
                                                : "strata-nh: %d NVFP4 matrices transcoded to Q8_0 (%.2f GiB in VRAM; no native NVFP4 MMVQ)\n",
                         nvfp4_mats, (double) nvfp4_bytes / (1 << 30));
    }
    m.out_norm = f32("output_norm.weight");
    {
        size_t s = 0;
        m.head = gm.find("output.weight", &s) ? mat("output.weight") : mat("token_embd.weight");
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        if (!K::embed_type_supported(m.embd_type)) { err = "token_embd's type is not one the embedding kernel takes"; return false; }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    m.dense_bytes = bytes;
    std::fprintf(stderr, "strata-nh: %s: %d sublayers (%d Mamba-2, %d attention, %d MoE)%s, %.2f GiB of dense weights on "
                 "the GPU in %.1f s (the routed experts stay in the mapped file)\n", path.c_str(), g.n_layers, m.n_mamba,
                 m.n_attn, m.n_moe, m.has_mtp ? " + the MTP block" : "", (double) bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the DFlash drafter
// Nemotron 3.5 Bolt's `dflash-*.gguf` (architecture `dflash` + `lilicorr.*`): upstream llama.cpp's model
// (src/models/dflash.cpp, common/speculative.cpp), re-done on this engine's kernels.
//   * context: the target's residual stream at the input of sublayers target_layers[k] (6 x 4096) -> fc -> RMS norm
//     (enc.output_norm) = g; per draft layer K = rope(k_norm(wk g)), V = wv g into a sliding-window (1024) ring;
//   * draft: the block [last token, MASK x 7] at positions pos .. pos+7 through 5 layers, each with the grouped
//     dynamic 2-tap convolutions (attn_conv / ffn_conv: in on the normed input, out on the sublayer's output), a
//     non-causal attention with sinks over [the window's context | the block], a SwiGLU FFN; output_norm, the
//     target's head;
//   * LiLiCorr: top-8 candidates of each of the 7 draft positions (nodes = 7 x 8) scored by a 2-layer transformer
//     over the nodes (token, draft-hidden, logit features, slot/rank embeddings) into pairwise transition scores;
//     the draft is the greedy walk over that lattice from the anchor (g of the last committed position).
// Matrices: NVFP4 (with .scale) stays NVFP4 under --nvfp4 native (the same MMVQ / dequant-GEMM path as the
// target's), else NVFP4 and BF16 -> Q8_0 at load (the native MMVQ / GEMM take Q8_0); LiLiCorr's F32 matrices
// stay F32 (f32mm).
std::vector<uint8_t> f32_to_q8_0(const float* x, int64_t n_in, int64_t n_out) {
    const int64_t nb = n_in / 32;
    std::vector<uint8_t> out((size_t) (n_out * nb * 34));
    for (int64_t r = 0; r < n_out; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            const float* v = x + r * n_in + b * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
            const float dq = amax / 127.f, id = dq != 0.f ? 1.f / dq : 0.f;
            uint8_t* q8 = out.data() + (size_t) ((r * nb + b) * 34);
            const __half hd = __float2half(dq);
            std::memcpy(q8, &hd, 2);
            for (int i = 0; i < 32; ++i) q8[2 + i] = (uint8_t) (int8_t) std::lrintf(v[i] * id);
        }
    return out;
}

struct DfLayer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *sinks, *attn_cbase, *ffn_cbase;
    Mat wq, wk, wv, wo, attn_conv, ffn_conv, gate, up, down;
};
struct LcLayer {
    const float *attn_norm, *mlp_norm, *in_proj, *in_proj_b, *out_b, *mlp0_b, *mlp2_b;
    Mat out_proj, mlp0, mlp2;
};
struct Dflash {
    int n_layers = 0, N = 0, H = 0, HKV = 0, FF = 0, block = 8, conv_k = 2, conv_g = 16, window = 1024, mask = 0;
    float eps = 1e-6f, rope_base = 10000.f;
    std::vector<int> taps;          // target sublayers whose input is tapped (fc's input order)
    std::vector<Mat> fc;            // fc split by tap: [k] = fc's columns k*N .. (k+1)*N
    const float *enc_norm = nullptr, *out_norm = nullptr;
    std::vector<DfLayer> L;
    // LiLiCorr
    int hidden = 0, topk = 0, heads = 0, lc_layers = 0, fdim = 0, mlp = 0;
    float vec_eps = 1e-4f, logit_scale = 1.f;
    Mat token_proj, pass_proj, ctx_proj;
    const float *token_b, *pass_b, *ctx_b, *fnorm_w, *fnorm_b, *f1_w, *f1_b, *f3_w, *f3_b, *slot_e, *rank_e, *abias;
    const float *out_norm_l, *anchor_norm, *fin_w, *fin_b, *out_w, *out_b, *in_w, *in_b, *aout_w, *aout_b;
    std::vector<LcLayer> LL;
    size_t bytes = 0;
};

bool load_dflash(const std::string& path, const Model& tm, Dflash& d, std::string& err) {
    const auto t0 = Clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(path);
    const strata::MetaValue* arch = meta(gm, "general.architecture");
    if (!arch || arch->s != "dflash") { err = "--dflash: not a dflash GGUF"; return false; }
    auto num = [&](const std::string& k, double dflt) {
        const strata::MetaValue* v = meta(gm, k);
        return v ? v->num() : dflt;
    };
    d.n_layers = (int) num("dflash.block_count", 0);
    d.N = (int) num("dflash.embedding_length", 0);
    d.H = (int) num("dflash.attention.head_count", 0);
    d.HKV = (int) num("dflash.attention.head_count_kv", 0);
    d.FF = (int) num("dflash.feed_forward_length", 0);
    d.block = (int) num("dflash.block_size", 0);
    d.conv_k = (int) num("dflash.conv_kernel_size", 0);
    d.conv_g = (int) num("dflash.conv_group_size", 0);
    d.window = (int) num("dflash.attention.sliding_window", 0);
    d.eps = (float) num("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    d.rope_base = (float) num("dflash.rope.freq_base", 10000.0);
    d.mask = (int) num("tokenizer.ggml.mask_token_id", -1);
    d.hidden = (int) num("lilicorr.hidden_size", 0);
    d.topk = (int) num("lilicorr.candidate_topk", 0);
    d.heads = (int) num("lilicorr.num_heads", 0);
    d.lc_layers = (int) num("lilicorr.num_layers", 0);
    d.fdim = (int) num("lilicorr.factor_dim", 0);
    d.mlp = (int) num("lilicorr.mlp_hidden_size", 0);
    d.vec_eps = (float) num("lilicorr.vector_eps", 1e-4);
    d.logit_scale = (float) num("lilicorr.logit_scale", 1.0);
    const int key_len = (int) num("dflash.attention.key_length", 0);
    const bool causal = num("dflash.attention.causal", 1) != 0;
    const strata::MetaValue* tl = meta(gm, "dflash.target_layers");
    if (tl && tl->type == strata::MetaType::ARRAY)
        for (const auto& it : tl->items) d.taps.push_back((int) it.num());
    // what this implementation covers (the Bolt drafter): LiLiCorr on, no DFlash2 selector, no DSpark head, the
    // SWA pattern all-sliding, a 2-tap conv, the hidden the target's
    const strata::MetaValue* swp = meta(gm, "dflash.attention.sliding_window_pattern");
    bool all_swa = swp && swp->type == strata::MetaType::ARRAY && (int) swp->items.size() == d.n_layers;
    if (all_swa) for (const auto& it : swp->items) all_swa &= it.num() != 0;
    if (d.N != tm.g.N || key_len != 128 || d.H * 128 != d.N || d.HKV <= 0 || d.H % d.HKV || d.block != 8 || d.conv_k < 1 ||
        d.conv_g <= 0 || d.N % d.conv_g || d.window <= 0 || d.window + 16 > kDfMaxKeys || causal || !all_swa ||
        d.taps.empty() || d.topk <= 0 || d.topk > 8 || d.hidden != d.heads * 128 || d.lc_layers <= 0 ||
        (d.block - 1) * d.topk > 64 || d.mask < 0 || num("dflash.selector_top_k", 0) != 0 || gm.find("markov_w1.weight") ||
        gm.find("d2t") || gm.find("output.weight") || gm.find("token_embd.weight")) {
        err = "--dflash: a drafter geometry this implementation does not take (it is written for Bolt's LiLiCorr "
              "drafter: block 8, head 128, all-SWA non-causal, LiLiCorr top-k <= 8, the target's embedding and head)";
        return false;
    }
    for (int t : d.taps)
        if (t < 0 || t >= tm.g.n_layers) { err = "--dflash: a target layer outside the model"; return false; }
    auto find = [&](const std::string& name) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name);
        if (!t) {
            std::fprintf(stderr, "strata-nh: --dflash: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) { std::fprintf(stderr, "strata-nh: --dflash: missing %s\n", name.c_str()); std::exit(1); }
        d.bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    // a matrix's rows as host bytes: NVFP4 kept NVFP4 under --nvfp4 native (its .scale on the output, as the main
    // model's), else Q8_0 (NVFP4 with its .scale folded in, BF16/F16/F32 quantized, Q8_0 as is)
    struct HostMat {
        std::vector<uint8_t> q;
        int type = kTypeQ8_0, n_in = 0, n_out = 0;
        float s2 = 1.f;
        const std::string* name = nullptr;
    };
    int nvfp4_native = 0;
    auto host_mat = [&](const std::string& name) -> HostMat {
        size_t s = 0;
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) { std::fprintf(stderr, "strata-nh: --dflash: missing %s\n", name.c_str()); std::exit(1); }
        HostMat h;
        h.n_in = (int) t->shape[0];
        h.n_out = (int) t->shape[1];
        if (h.n_in % 64) { std::fprintf(stderr, "strata-nh: --dflash: %s has rows of %d\n", name.c_str(), h.n_in); std::exit(1); }
        const std::string base = name.substr(0, name.size() - 7);   // ".weight"
        float scale2 = 1.f;
        size_t ss = 0;
        if (const strata::TensorInfo* ts = gm.find(base + ".scale", &ss)) {
            if (ts->elements() != 1 || ts->type != 0) { std::fprintf(stderr, "strata-nh: %s.scale is not one F32\n", base.c_str()); std::exit(1); }
            std::memcpy(&scale2, gm.shard(ss).tensor_data(*ts), 4);
            if (t->type != kTypeNvfp4) { std::fprintf(stderr, "strata-nh: %s: a .scale on a non-NVFP4 matrix\n", name.c_str()); std::exit(1); }
        }
        const uint8_t* src = gm.shard(s).tensor_data(*t);
        if (t->type == kTypeNvfp4 && g_nvfp4_native) {
            if (h.n_in % 256) { std::fprintf(stderr, "strata-nh: --dflash: %s is NVFP4 with a row of %d (not whole 256s)\n", name.c_str(), h.n_in); std::exit(1); }
            h.q.assign(src, src + strata::tensor_payload_bytes(*t));
            h.type = kTypeNvfp4;
            h.s2 = scale2;
            ++nvfp4_native;
            return h;
        }
        if (t->type == kTypeNvfp4) { h.q = nvfp4_to_q8_0(src, h.n_in, h.n_out, scale2); return h; }
        if (t->type == kTypeQ8_0) { h.q.assign(src, src + strata::tensor_payload_bytes(*t)); return h; }
        std::vector<float> f((size_t) h.n_in * h.n_out);
        if (t->type == 0) std::memcpy(f.data(), src, f.size() * 4);
        else if (t->type == 30) for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
        else if (t->type == 1) for (size_t i = 0; i < f.size(); ++i) f[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
        else { std::fprintf(stderr, "strata-nh: --dflash: %s is %s\n", name.c_str(), t->type_name()); std::exit(1); }
        h.q = f32_to_q8_0(f.data(), h.n_in, h.n_out);
        return h;
    };
    auto up = [&](const HostMat& h, const std::string& name) -> Mat {
        Mat w;
        w.type = h.type;
        w.n_in = h.n_in;
        w.n_out = h.n_out;
        w.s2 = h.s2;
        const size_t tail = h.type == kTypeNvfp4 ? kTail : 0;   // the NVFP4 MMVQ reads a little past a matrix
        uint8_t* p = nullptr;
        CK(cudaMalloc(&p, h.q.size() + tail));
        if (tail) CK(cudaMemset(p + h.q.size(), 0, tail));
        CK(cudaMemcpy(p, h.q.data(), h.q.size(), cudaMemcpyHostToDevice));
        if (h.type == kTypeNvfp4 && std::getenv("STRATA_NH_NVFP4_CHECK")) nvfp4_check("dflash " + name, h.q.data(), p, h.n_in, h.n_out, h.s2);
        d.bytes += h.q.size();
        w.d = p;
        return w;
    };
    auto mat = [&](const std::string& name) -> Mat { return up(host_mat(name), name); };
    {   // fc [taps * N -> N], split by tap along the rows' blocks (Q8_0: 34 bytes of 32; NVFP4: 36 bytes of 64): the
        // slice of tap k is blocks k*N/bv .. (k+1)*N/bv of every row
        const HostMat h = host_mat("fc.weight");
        if (h.n_in != (int) d.taps.size() * d.N || h.n_out != d.N) { err = "--dflash: fc.weight is not [taps x N -> N]"; return false; }
        const size_t bv = h.type == kTypeNvfp4 ? 64 : 32, bb = h.type == kTypeNvfp4 ? 36 : 34;
        const size_t row = (size_t) h.n_in / bv * bb, part = (size_t) d.N / bv * bb;
        for (size_t k = 0; k < d.taps.size(); ++k) {
            HostMat s;
            s.type = h.type;
            s.n_in = d.N;
            s.n_out = h.n_out;
            s.s2 = h.s2;
            s.q.resize((size_t) h.n_out * part);
            for (int r = 0; r < h.n_out; ++r) std::memcpy(s.q.data() + (size_t) r * part, h.q.data() + (size_t) r * row + k * part, part);
            d.fc.push_back(up(s, "fc.weight[" + std::to_string(k) + "]"));
        }
    }
    d.enc_norm = f32("enc.output_norm.weight");
    d.out_norm = f32("output_norm.weight");
    auto B = [](int l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    for (int l = 0; l < d.n_layers; ++l) {
        DfLayer y;
        y.attn_norm = f32(B(l, "attn_norm.weight"));
        y.ffn_norm = f32(B(l, "ffn_norm.weight"));
        y.q_norm = f32(B(l, "attn_q_norm.weight"));
        y.k_norm = f32(B(l, "attn_k_norm.weight"));
        y.sinks = f32(B(l, "attn_sinks.weight"));
        y.attn_cbase = f32(B(l, "attn_conv_base"));
        y.ffn_cbase = f32(B(l, "ffn_conv_base"));
        y.wq = mat(B(l, "attn_q.weight"));
        y.wk = mat(B(l, "attn_k.weight"));
        y.wv = mat(B(l, "attn_v.weight"));
        y.wo = mat(B(l, "attn_output.weight"));
        y.attn_conv = mat(B(l, "attn_conv_proj.weight"));
        y.ffn_conv = mat(B(l, "ffn_conv_proj.weight"));
        y.gate = mat(B(l, "ffn_gate.weight"));
        y.up = mat(B(l, "ffn_up.weight"));
        y.down = mat(B(l, "ffn_down.weight"));
        const int proj = 2 * d.conv_k * (d.N / d.conv_g);
        const strata::TensorInfo* cb = find(B(l, "attn_conv_base"));
        if (y.wq.n_out != d.N || y.wk.n_out != d.HKV * 128 || y.wv.n_out != d.HKV * 128 || y.attn_conv.n_out != proj ||
            y.ffn_conv.n_out != proj || y.gate.n_out != d.FF || y.down.n_in != d.FF || cb->elements() != (uint64_t) 2 * d.conv_k * d.N) {
            err = "--dflash: " + B(l, "tensors have unexpected shapes");
            return false;
        }
        d.L.push_back(y);
    }
    // LiLiCorr: its blocks are blk.<n_layers + i>, the feature MLP blk.<n_layers + lc_layers>
    const int fb = d.n_layers + d.lc_layers;
    d.token_proj = mat("lilicorr.token_proj.weight");
    d.token_b = f32("lilicorr.token_proj.bias");
    d.pass_proj = mat("lilicorr.pass_hidden_proj.weight");
    d.pass_b = f32("lilicorr.pass_hidden_proj.bias");
    d.ctx_proj = mat("lilicorr.context_proj.weight");
    d.ctx_b = f32("lilicorr.context_proj.bias");
    d.fnorm_w = f32(B(fb, "ffn_norm.weight"));
    d.fnorm_b = f32(B(fb, "ffn_norm.bias"));
    d.f1_w = f32(B(fb, "ffn_up.weight"));
    d.f1_b = f32(B(fb, "ffn_up.bias"));
    d.f3_w = f32(B(fb, "ffn_down.weight"));
    d.f3_b = f32(B(fb, "ffn_down.bias"));
    d.slot_e = f32("lilicorr.slot_embedding.weight");
    d.rank_e = f32("lilicorr.rank_embedding.weight");
    d.abias = f32("lilicorr.attention_bias.bias");
    d.out_norm_l = f32("lilicorr.output_norm.weight");
    d.anchor_norm = f32("lilicorr.anchor_norm.weight");
    d.fin_w = f32("lilicorr.factor_input_proj.weight");
    d.fin_b = f32("lilicorr.factor_input_proj.bias");
    d.out_w = f32("lilicorr.out_head.weight");
    d.out_b = f32("lilicorr.out_head.bias");
    d.in_w = f32("lilicorr.in_head.weight");
    d.in_b = f32("lilicorr.in_head.bias");
    d.aout_w = f32("lilicorr.anchor_out_head.weight");
    d.aout_b = f32("lilicorr.anchor_out_head.bias");
    {
        const int nodes = (d.block - 1) * d.topk;
        const strata::TensorInfo* ab = find("lilicorr.attention_bias.bias");
        const strata::TensorInfo* f1 = find(B(fb, "ffn_up.weight"));
        const strata::TensorInfo* ow = find("lilicorr.out_head.weight");
        if (ab->elements() != (uint64_t) nodes * nodes * d.heads || f1->shape[0] != 5 || (int) f1->shape[1] != d.hidden ||
            (int) ow->shape[1] != d.fdim || d.token_proj.n_out != d.hidden || d.pass_proj.n_in != d.N ||
            find("lilicorr.slot_embedding.weight")->elements() != (uint64_t) (d.block - 1) * d.hidden ||
            find("lilicorr.rank_embedding.weight")->elements() != (uint64_t) d.topk * d.hidden ||
            find("lilicorr.factor_input_proj.weight")->shape[0] != (uint64_t) 3 * d.hidden) {
            err = "--dflash: LiLiCorr tensors have unexpected shapes";
            return false;
        }
    }
    for (int i = 0; i < d.lc_layers; ++i) {
        const int b = d.n_layers + i;
        LcLayer y;
        y.attn_norm = f32(B(b, "attn_norm.weight"));
        y.mlp_norm = f32(B(b, "ffn_norm.weight"));
        y.in_proj = f32("lilicorr.layers." + std::to_string(i) + ".attn.in_proj.weight");
        y.in_proj_b = f32("lilicorr.layers." + std::to_string(i) + ".attn.in_proj.bias");
        y.out_proj = mat(B(b, "attn_output.weight"));
        y.out_b = f32(B(b, "attn_output.bias"));
        y.mlp0 = mat(B(b, "ffn_up.weight"));
        y.mlp0_b = f32(B(b, "ffn_up.bias"));
        y.mlp2 = mat(B(b, "ffn_down.weight"));
        y.mlp2_b = f32(B(b, "ffn_down.bias"));
        if (y.mlp0.n_out != d.mlp || y.out_proj.n_in != d.hidden) { err = "--dflash: LiLiCorr layer shapes"; return false; }
        d.LL.push_back(y);
    }
    std::fprintf(stderr, "strata-nh: DFlash drafter %s: %d layers, block %d, taps", path.c_str(), d.n_layers, d.block);
    for (int t : d.taps) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, ", window %d, mask %d; LiLiCorr top-%d x %d layers; %.2f GiB on the GPU (%s) in %.1f s\n", d.window,
                 d.mask, d.topk, d.lc_layers, (double) d.bytes / (1 << 30),
                 nvfp4_native ? "NVFP4 kept native + Q8_0 for the rest" : "Q8_0", ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert stream
/// The prompt path's expert stream: the non-resident experts of the forward's MoE layers, in order, through a ring
/// of `nr` VRAM buffers.  A copy thread reads each expert from the mapped GGUF (into a pinned bounce slot, or
/// straight when the mapping is registered with --pin-experts) and queues its upload on a copy stream; the engine's
/// stream waits for an item's upload before its GEMMs and records when it is done with the buffer.
struct ExpertStream {
    struct Item {
        const uint8_t *up, *down;
        size_t up_bytes, d_bytes;
    };
    int nr = 0;
    size_t slot = 0;
    uint8_t* ring = nullptr;     // device, nr * slot
    uint8_t* bounce = nullptr;   // pinned, nr * slot (null: the sources are registered)
    cudaStream_t cst = nullptr;
    std::vector<cudaEvent_t> ready, freed;
    std::vector<Item> plan;
    std::atomic<long> enq{0}, done_n{0};
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false;

    void init(int n, size_t slot_bytes, bool registered) {
        nr = n;
        slot = slot_bytes;
        CK(cudaMalloc(&ring, (size_t) nr * slot + kTail));
        CK(cudaMemset(ring, 0, (size_t) nr * slot + kTail));
        if (!registered) CK(cudaHostAlloc((void**) &bounce, (size_t) nr * slot, cudaHostAllocDefault));
        CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        ready.resize((size_t) nr);
        freed.resize((size_t) nr);
        for (int i = 0; i < nr; ++i) {
            CK(cudaEventCreateWithFlags(&ready[(size_t) i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&freed[(size_t) i], cudaEventDisableTiming));
        }
    }
    void begin(std::vector<Item> items) {
        finish();
        plan = std::move(items);
        enq = 0;
        done_n = 0;
        stop = false;
        if (plan.empty()) return;
        th = std::thread([this] { copier(); });
    }
    void copier() {
        CK(cudaSetDevice(dev));
        for (long i = 0; i < (long) plan.size(); ++i) {
            const int s = (int) (i % nr);
            if (i >= nr) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || done_n.load() > i - nr; });
                if (stop) return;
                lk.unlock();
                CK(cudaEventSynchronize(freed[(size_t) s]));
            }
            const Item& it = plan[(size_t) i];
            uint8_t* dst = ring + (size_t) s * slot;
            if (bounce) {
                uint8_t* b = bounce + (size_t) s * slot;
                std::memcpy(b, it.up, it.up_bytes);
                std::memcpy(b + it.up_bytes, it.down, it.d_bytes);
                CK(cudaMemcpyAsync(dst, b, it.up_bytes + it.d_bytes, cudaMemcpyHostToDevice, cst));
            } else if (it.down == it.up + it.up_bytes) {   // the arena's blob: one copy
                CK(cudaMemcpyAsync(dst, it.up, it.up_bytes + it.d_bytes, cudaMemcpyHostToDevice, cst));
            } else {
                CK(cudaMemcpyAsync(dst, it.up, it.up_bytes, cudaMemcpyHostToDevice, cst));
                CK(cudaMemcpyAsync(dst + it.up_bytes, it.down, it.d_bytes, cudaMemcpyHostToDevice, cst));
            }
            CK(cudaEventRecord(ready[(size_t) s], cst));
            {
                std::lock_guard<std::mutex> lk(mu);
                enq = i + 1;
            }
            cv.notify_all();
        }
    }
    /// item i's buffer, once its upload is queued (the engine's stream waits for it)
    const uint8_t* take(long i, cudaStream_t st) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return enq.load() > i; });
        }
        const int s = (int) (i % nr);
        CK(cudaStreamWaitEvent(st, ready[(size_t) s], 0));
        return ring + (size_t) s * slot;
    }
    void done(long i, cudaStream_t st) {
        CK(cudaEventRecord(freed[(size_t) (i % nr)], st));
        {
            std::lock_guard<std::mutex> lk(mu);
            done_n = i + 1;
        }
        cv.notify_all();
    }
    void finish() {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        th.join();
    }
    int dev = 0;
};

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * 4));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;
    static constexpr int kVerify = 8;
    static constexpr int kGemmT = 8;   // a batch above this many rows runs its dense projections as GEMMs
    // activations
    float *x, *h, *y, *y2, *zx, *xc, *yr, *yg, *qx, *kx, *vx, *att, *logits_r, *ewts, *hl, *lat, *eparts, *su, *logits;
    float *hmain, *hlast, *seeds, *hmtp, *mtp_cat, *logits_m;
    float *ubuf = nullptr;
    __half *x16 = nullptr, *xg16 = nullptr, *hb16 = nullptr, *q16 = nullptr, *att_p16 = nullptr;
    float* att_s = nullptr;
    size_t kAttScoreFloats = (size_t) 32 << 20;
    void *q8, *xq, *escr;
    void* xq_l = nullptr;   // the latent rows' q8_1 image: the routed experts' input (every grouped call of a layer)
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick, *tok_m, *pos0_m, *pick_m;
    int32_t *rcnt, *rcur, *rbounds, *rinv;
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int32_t* hpin = nullptr;   // pinned staging: [0] token, [1] position, [2..9] picks, [16..] tokens
    int n_chunk = 0, att_rows = 0;
    std::vector<float*> conv, S, vbuf;   // per Mamba layer: conv history, state, a verify window's zx rows
    std::vector<__half*> kc, vc;         // per attention layer (+ the MTP block's at n_attn)
    // --kv q8_0 / q4_0 (kv_fmt 1 / 2): codes + per-32 FP16 scales instead of kc / vc (nh_kernels.cuh); the prompt's
    // GEMMs read FP16 slices of kKvSlice cells dequantized into kv_tk / kv_tv.  0 (default): the FP16 cache above.
    int kv_fmt = 0;
    std::vector<uint8_t*> kq, vq;
    std::vector<__half*> ks, vs;
    static constexpr int kKvSlice = 16384;
    __half *kv_tk = nullptr, *kv_tv = nullptr;
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;
    int mtp_d = 0;
    bool prompt_mtp = true;   // the prompt also runs through the MTP block (its KV) when drafts are on
    // the hybrid mode
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance (the slots follow it)
        std::vector<double> total;   // never halved: the expert profile this engine saves (--expert-profile-save)
        int32_t* slot_of_d = nullptr;
    };
    // ---- transient GPU experts, Flash-Next's two ways of not computing a miss on the CPU: the PCIe share (a part of
    // a layer's misses uploaded when the routing is known, while the CPU computes the rest) and the prefetch (the next
    // MoE layer's likely experts, from its router applied to the hidden one sublayer early, uploaded while that
    // sublayer runs on the GPU and the CPU is idle).  Both into small VRAM buffers, run by one grouped call.
    float pcie_frac = 0.f;
    int pcie_cap = 48, pf_cap = 0;
    size_t tr_stride = 0;
    uint8_t* pcbuf = nullptr;
    uint8_t* pfbuf[2] = {};
    cudaStream_t cst2 = nullptr;
    cudaEvent_t ev_pc = nullptr, ev_trfree = nullptr, ev_pfroute = nullptr, ev_pf[2] = {}, ev_pffree[2] = {};
    unsigned long long* h_tgp = nullptr;
    int32_t *h_tgs = nullptr, *h_ted = nullptr, *h_tet = nullptr, *h_tng = nullptr;
    unsigned long long* d_tgp = nullptr;
    int32_t *d_tgs = nullptr, *d_ted = nullptr, *d_tet = nullptr, *d_tng = nullptr;
    float *logits_pf = nullptr, *wts_pf = nullptr;
    int32_t *ids_pf = nullptr, *h_pfids = nullptr;
    struct Pf {
        int mi = -1;                 // the MoE slot index whose prediction the buffer holds (-1: none)
        std::vector<int> slot;       // expert -> its place in the buffer (-1: not there)
    } pf[2];
    long long tr_pcie = 0, tr_pf_sent = 0, tr_pf_hit = 0;
    std::string profile_save;   // --expert-profile-save: the routing counts saved here (every 8 requests, at exit)
    // ---- turn checkpoints (Flash-Next's conversation checkpoints, reduced to what a recurrent state needs): the
    // Mamba states (conv history + scan state) and hlast at the end of a request's prompt, in pinned host memory.  The
    // state cannot step back, but the KV cache can (its cells past a position are just overwritten), so a request that
    // goes back to a checkpoint along the live sequence (the client regenerating or editing the last answer) restores
    // it and reads only the rest of its prompt.
    struct Ckpt {
        int64_t pos = 0;           // the tokens it holds: live[0 .. pos)
        float* host = nullptr;
        uint64_t used = 0;
    };
    std::vector<Ckpt> ckpts;
    int max_ckpts = 4;
    uint64_t ckpt_clock = 0;
    size_t ckpt_floats() const {
        const Geo& g = m->g;
        return (size_t) m->n_mamba * ((size_t) g.conv_ch * (g.ssm_conv - 1) + (size_t) g.ssm_H * kSsmP * kSsmN) + g.N;
    }
    void ckpt_copy(float* host, bool save) {
        const Geo& g = m->g;
        const size_t nc = (size_t) g.conv_ch * (g.ssm_conv - 1), ns = (size_t) g.ssm_H * kSsmP * kSsmN;
        size_t off = 0;
        const auto kind = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        for (int i = 0; i < m->n_mamba; ++i) {
            if (save) {
                CK(cudaMemcpyAsync(host + off, conv[(size_t) i], nc * 4, kind, st));
                CK(cudaMemcpyAsync(host + off + nc, S[(size_t) i], ns * 4, kind, st));
            } else {
                CK(cudaMemcpyAsync(conv[(size_t) i], host + off, nc * 4, kind, st));
                CK(cudaMemcpyAsync(S[(size_t) i], host + off + nc, ns * 4, kind, st));
            }
            off += nc + ns;
        }
        if (save) CK(cudaMemcpyAsync(host + off, hlast, (size_t) g.N * 4, kind, st));
        else CK(cudaMemcpyAsync(hlast, host + off, (size_t) g.N * 4, kind, st));
        CK(cudaStreamSynchronize(st));
    }
    /// A checkpoint of the current state (pos tokens); an existing one at the same position is refreshed.
    void ckpt_save() {
        if (max_ckpts <= 0 || pos <= 0) return;
        Ckpt* c = nullptr;
        for (Ckpt& k : ckpts) if (k.pos == pos) c = &k;
        if (!c && (int) ckpts.size() < max_ckpts) {
            ckpts.push_back({});
            c = &ckpts.back();
            if (cudaHostAlloc((void**) &c->host, ckpt_floats() * 4, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                ckpts.pop_back();
                return;
            }
        }
        if (!c) {   // the least recently used one
            c = &ckpts[0];
            for (Ckpt& k : ckpts) if (k.used < c->used) c = &k;
        }
        c->pos = pos;
        c->used = ++ckpt_clock;
        ckpt_copy(c->host, true);
    }
    /// The checkpoints past `keep` tokens no longer describe the live sequence (its cells were rewritten from there).
    void ckpt_drop_past(int64_t keep) {
        for (Ckpt& k : ckpts) if (k.pos > keep) k.pos = -1;
    }
    /// The longest checkpoint at or below `limit` tokens: restored, its position returned (0: none).
    int64_t ckpt_restore(int64_t limit) {
        Ckpt* best = nullptr;
        for (Ckpt& k : ckpts) if (k.pos > 0 && k.pos <= limit && (!best || k.pos > best->pos)) best = &k;
        if (!best) return 0;
        ckpt_copy(best->host, false);
        best->used = ++ckpt_clock;
        pos = best->pos;
        return pos;
    }
    double t_route_wait_ms = 0;
    void init_transient() {
        const Geo& g = m->g;
        const size_t ent = (size_t) std::max(std::min(hybrid_t, tmax), kVerify) * g.top_k;
        CK(cudaStreamCreateWithFlags(&cst2, cudaStreamNonBlocking));
        for (cudaEvent_t* e : {&ev_pc, &ev_trfree, &ev_pf[0], &ev_pf[1], &ev_pffree[0], &ev_pffree[1]})
            CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_pfroute, cudaEventDisableTiming | cudaEventBlockingSync));
        CK(cudaHostAlloc((void**) &h_tgp, (ent + 1) * 8, cudaHostAllocDefault));
        for (int32_t** b : {&h_tgs, &h_ted, &h_tet, &h_tng}) CK(cudaHostAlloc((void**) b, (ent + 1) * 4, cudaHostAllocDefault));
        d_tgp = dalloc_t<unsigned long long>(ent + 1);
        d_tgs = dalloc_t<int32_t>(ent + 1); d_ted = dalloc_t<int32_t>(ent + 1); d_tet = dalloc_t<int32_t>(ent + 1);
        d_tng = dalloc_t<int32_t>(1);
        const size_t rows = (size_t) std::max(std::min(hybrid_t, tmax), kVerify);
        logits_pf = dalloc(rows * g.E); wts_pf = dalloc(rows * g.top_k); ids_pf = dalloc_t<int32_t>(rows * g.top_k);
        CK(cudaHostAlloc((void**) &h_pfids, rows * g.top_k * 4, cudaHostAllocDefault));
        for (Pf& P : pf) P.slot.assign((size_t) g.E, -1);
    }
    /// The next MoE layer's routing guessed from the current hidden (before the sublayer between runs): queued now,
    /// read by prefetch_issue once that sublayer's kernels are queued too.
    void prefetch_begin(const Layer& Ln, int T) {
        const Geo& g = m->g;
        f32mm(Ln.moe.router, h, logits_pf, g.N, Ln.moe.layer_e, T);
        k_route<<<T, 512, 0, st>>>(logits_pf, Ln.moe.bias, Ln.moe.layer_e, g.top_k, g.w_scale, g.w_norm, ids_pf, wts_pf);
        CK(cudaMemcpyAsync(h_pfids, ids_pf, (size_t) T * g.top_k * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_pfroute, st));
    }
    void prefetch_issue(const Layer& Ln, int T) {
        const Geo& g = m->g;
        const Moe& M = Ln.moe;
        const int mi = Ln.idx, b = mi & 1;
        CK(cudaEventSynchronize(ev_pfroute));
        Pf& P = pf[b];
        for (int x = 0; x < g.E; ++x) P.slot[(size_t) x] = -1;
        P.mi = mi;
        const Hy& Hh = hy[(size_t) mi];
        CK(cudaStreamWaitEvent(cst2, ev_pffree[b], 0));
        int n = 0;
        // rank order across the rows: every row's first pick, then every row's second, ...
        for (int k = 0; k < g.top_k && n < pf_cap; ++k)
            for (int t = 0; t < T && n < pf_cap; ++t) {
                const int x = h_pfids[t * g.top_k + k];
                if (Hh.slot_of[(size_t) x] >= 0 || P.slot[(size_t) x] >= 0) continue;
                P.slot[(size_t) x] = n;
                CK(cudaMemcpyAsync(pfbuf[b] + (size_t) n * tr_stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice, cst2));
                CK(cudaMemcpyAsync(pfbuf[b] + (size_t) n * tr_stride + M.L.down_off, M.down_of(x), M.d_bytes,
                                   cudaMemcpyHostToDevice, cst2));
                ++n;
            }
        if (!warming) tr_pf_sent += n;
        CK(cudaEventRecord(ev_pf[b], cst2));
    }
    std::vector<Hy> hy;   // per MoE slot index (the main layers, then the MTP block)
    int hybrid_t = 64;
    std::unique_ptr<strata::nemotron_h::CpuExperts> pool;
    std::vector<strata::nemotron_h::CpuJob> jobs;
    float *h_hl = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    cudaEvent_t ev_route = nullptr;
    // STRATA_NH_SKIP_THR=w / STRATA_NH_SKIP_REL=r: a CPU-side entry (no VRAM slot, not prefetched, not in the PCIe share) whose
    // routing weight is < w, or < r * its token row's largest weight, is not computed (its row stays zero); with
    // STRATA_NH_SKIP_RENORM=1 the kept weights of the row are scaled back to the row's old sum.  The weights are sigmoid
    // probabilities normalized to sum 1 (g.w_norm) times g.w_scale.  Rows paths only (decode, MTP/DFlash verify).
    float skip_thr = std::getenv("STRATA_NH_SKIP_THR") ? (float) std::atof(std::getenv("STRATA_NH_SKIP_THR")) : 0.f;
    float skip_rel = std::getenv("STRATA_NH_SKIP_REL") ? (float) std::atof(std::getenv("STRATA_NH_SKIP_REL")) : 0.f;
    bool skip_renorm = std::getenv("STRATA_NH_SKIP_RENORM") != nullptr && std::string(std::getenv("STRATA_NH_SKIP_RENORM")) != "0";
    float *h_w = nullptr, *h_w2 = nullptr;   // the routing weights on the host (pinned), the renormalized ones
    long long skip_n = 0, skip_tot = 0;
    ExpertStream xs;
    long xs_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0, hy_streamed = 0;
    double hy_cpu_ms = 0;
    bool warming = false;
    int adapt_every = 32, adapt_swaps = 8;   // --adapt-every / --adapt-swaps: the decode's slot replacements
    int swaps_after_prompt = 512;
    // GENI: rows that replace the embedding of an absolute position (host floats, N each)
    std::map<int64_t, const float*> inject;
    // --dump-layers: every main forward's sublayer outputs (the residual x), [pos][layer][N] f32 (parity.py)
    FILE* dump_f = nullptr;
    std::vector<float> dump_buf;
    // ---- the DFlash drafter (--dflash; see load_dflash): the context taps ride on every main forward, a step drafts
    // a block and verifies it with the same window as --mtp (the Mamba state committed for the accepted rows only)
    const Dflash* df = nullptr;
    int df_n = 0, df_R = 0;            // drafts per step (<= block - 1); ring cells (>= window + block, >= the chunk)
    std::vector<int> df_tap;           // per target sublayer: its index among the taps (-1: none)
    std::vector<__half*> df_kr, df_vr; // per draft layer: the context K / V ring [cell][HKV][128]
    float *df_acc = nullptr, *df_g = nullptr, *df_k = nullptr, *df_v = nullptr, *df_anchor = nullptr;
    float *dx_ = nullptr, *dh_ = nullptr, *dh2_ = nullptr, *ddyn = nullptr, *dq_ = nullptr, *dk_ = nullptr, *dv_ = nullptr;
    float *dat = nullptr, *dy_ = nullptr, *dy2_ = nullptr, *dg_ = nullptr, *du_ = nullptr, *dhout = nullptr, *dlog = nullptr;
    void* dq8 = nullptr;
    __half* dx16 = nullptr;
    int32_t *dtok = nullptr, *dcand = nullptr;
    float* dfeat = nullptr;
    float *lc_emb = nullptr, *lc_tok = nullptr, *lc_pass = nullptr, *lc_f0 = nullptr, *lc_f = nullptr, *lc_x = nullptr,
          *lc_a = nullptr, *lc_qkv = nullptr, *lc_ao = nullptr, *lc_t = nullptr, *lc_m0 = nullptr, *lc_anc = nullptr,
          *lc_ancn = nullptr, *lc_fin = nullptr, *lc_fac = nullptr, *lc_out = nullptr, *lc_in = nullptr, *lc_aout = nullptr,
          *lc_sc = nullptr;
    float *h_dlog = nullptr, *h_sc = nullptr, *h_feat = nullptr;
    int32_t* h_cand = nullptr;
    long long df_offered = 0, df_accepted = 0, df_steps = 0;
    double df_draft_ms = 0, df_verify_ms = 0;
    size_t df_vram = 0;

    void init_dflash() {
        const Dflash& D = *df;
        const Geo& g = m->g;
        size_t fr0 = 0, fr1 = 0, tot = 0;
        CK(cudaDeviceSynchronize());
        CK(cudaMemGetInfo(&fr0, &tot));
        const size_t T = (size_t) tmax, N = (size_t) g.N, B = (size_t) D.block, KV = (size_t) D.HKV * 128;
        const size_t nodes = (B - 1) * D.topk, hid = (size_t) D.hidden, proj = (size_t) 2 * D.conv_k * (D.N / D.conv_g);
        df_R = std::max(tmax, D.window + 16);
        df_tap.assign((size_t) g.n_layers, -1);
        for (size_t k = 0; k < D.taps.size(); ++k) df_tap[(size_t) D.taps[k]] = (int) k;
        for (int l = 0; l < D.n_layers; ++l) {
            df_kr.push_back(dalloc_t<__half>((size_t) df_R * KV));
            df_vr.push_back(dalloc_t<__half>((size_t) df_R * KV));
        }
        df_acc = dalloc(T * N); df_g = dalloc(T * N); df_k = dalloc(T * KV); df_v = dalloc(T * KV); df_anchor = dalloc(N);
        for (float** b : {&dx_, &dh_, &dh2_, &dq_, &dat, &dy_, &dy2_, &dhout}) *b = dalloc(B * N);
        ddyn = dalloc(B * proj); dk_ = dalloc(B * KV); dv_ = dalloc(B * KV);
        dg_ = dalloc(B * D.FF); du_ = dalloc(B * D.FF); dlog = dalloc(B * g.V);
        CK(cudaMalloc(&dq8, K::native_q8_1_bytes(std::max(D.FF, D.N), 1) * B));
        dx16 = dalloc_t<__half>(std::max<size_t>(T, 64) * std::max<size_t>(N, (size_t) D.mlp));
        dtok = dalloc_t<int32_t>(B); dcand = dalloc_t<int32_t>(nodes); dfeat = dalloc(nodes * 5);
        lc_emb = dalloc(nodes * N);
        for (float** b : {&lc_tok, &lc_f0, &lc_f, &lc_x, &lc_a, &lc_ao, &lc_t, &lc_fac}) *b = dalloc(nodes * hid);
        lc_pass = dalloc(B * hid); lc_qkv = dalloc(nodes * 3 * hid); lc_fin = dalloc(nodes * 3 * hid);
        lc_m0 = dalloc(nodes * D.mlp); lc_anc = dalloc(hid); lc_ancn = dalloc(hid);
        lc_out = dalloc(nodes * D.fdim); lc_in = dalloc(nodes * D.fdim); lc_aout = dalloc(D.fdim);
        lc_sc = dalloc(64 * 64);
        CK(cudaHostAlloc((void**) &h_dlog, (B - 1) * g.V * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_sc, 64 * 64 * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_feat, nodes * 5 * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_cand, nodes * 4, cudaHostAllocDefault));
        CK(cudaDeviceSynchronize());
        CK(cudaMemGetInfo(&fr1, &tot));
        df_vram = D.bytes + (fr0 - fr1);
        std::fprintf(stderr, "strata-nh: DFlash: %d drafts a step, ring %d cells; %.2f GiB of VRAM (weights %.2f + buffers %.2f)\n",
                     df_n, df_R, (double) df_vram / (1 << 30), (double) D.bytes / (1 << 30), (double) (fr0 - fr1) / (1 << 30));
    }
    /// out [T][n_out] = w x on the drafter's own scratch (its rows are wider than the engine's q8 / x16)
    void dmm(const Mat& w, const float* xin, float* out, int T) {
        if (T > kGemmT) {
            k_to_f16<<<grid1((int64_t) w.n_in * T), kThreads, 0, st>>>(xin, dx16, (int64_t) w.n_in * T);
            if (w.type == kTypeNvfp4) {   // --nvfp4 native: as apply(), NVFP4 -> FP16 in the GEMM's scratch, then .scale
                const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(w.n_out, gemm.scratch_elems() / w.n_in));
                const size_t rb = (size_t) w.n_in / 64 * 36;
                for (int64_t r0 = 0; r0 < w.n_out; r0 += rows) {
                    const int64_t n = std::min<int64_t>(rows, w.n_out - r0);
                    K::iq_dequant_f16(kTypeNvfp4, w.d + (size_t) r0 * rb, n * w.n_in, gemm.scratch(), st);
                    gemm.f16((const uint16_t*) dx16, gemm.scratch(), out + r0, T, n, w.n_in, w.n_out);
                }
                k_scale_f32<<<grid1((int64_t) T * w.n_out), kThreads, 0, st>>>(out, (int64_t) T * w.n_out, w.s2);
                return;
            }
            gemm.native((const uint16_t*) dx16, w.type, w.d, out, T, w.n_out, w.n_in);
        } else {
            K::native_quantize_q8_1(xin, dq8, w.n_in * T, 1, st);
            mmq(w, dq8, out, T);
        }
    }
    void bias(float* xp, const float* b, int n, int T, int act = 0) {
        k_df_bias_act<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xp, b, n, (int64_t) n * T, act);
    }
    /// A main forward's tap: x (the residual at the input of sublayer l) through fc's slice, summed into df_acc.
    void df_tap_layer(size_t l, int T) {
        const int N = m->g.N;
        if (l == 0) CK(cudaMemsetAsync(df_acc, 0, (size_t) T * N * 4, st));
        const int k = df_tap[l];
        if (k < 0) return;
        dmm(df->fc[(size_t) k], x, df_g, T);
        k_df_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(df_acc, df_g, (int64_t) T * N);
    }
    /// After a main forward over positions p0 .. p0+T-1: g = norm(fc(taps)), every draft layer's K / V into the ring.
    void df_context(int T, int64_t p0) {
        const Dflash& D = *df;
        const int N = m->g.N, KV = D.HKV * 128;
        k_rms_ld<<<T, kThreads, 0, st>>>(df_acc, N, D.enc_norm, df_g, N, N, D.eps);
        for (int l = 0; l < D.n_layers; ++l) {
            const DfLayer& y = D.L[(size_t) l];
            dmm(y.wk, df_g, df_k, T);
            dmm(y.wv, df_g, df_v, T);
            k_df_norm_rope<<<dim3(T, D.HKV), 128, 0, st>>>(df_k, KV, y.k_norm, D.eps, p0, D.rope_base);
            k_df_ring_store<<<dim3(T, D.HKV), 128, 0, st>>>(df_k, df_v, df_kr[(size_t) l], df_vr[(size_t) l], D.HKV, p0, df_R);
        }
    }
    /// The block [x_tok, MASK ...] at positions pos .. pos+block-1 -> n drafts (the LiLiCorr lattice's greedy walk).
    void df_draft(int x_tok, int n, std::vector<int32_t>& drafts) {
        const Dflash& D = *df;
        const Geo& g = m->g;
        const int N = g.N, B = D.block, KV = D.HKV * 128, proj = 2 * D.conv_k * (D.N / D.conv_g), S = B - 1;
        const int nodes = S * D.topk, hid = D.hidden, tk = D.topk;
        const float scale = 1.0f / sqrtf(128.f);
        CK(cudaStreamSynchronize(st));
        hpin[16] = x_tok;
        for (int i = 1; i < B; ++i) hpin[16 + i] = D.mask;
        CK(cudaMemcpyAsync(dtok, hpin + 16, (size_t) B * 4, cudaMemcpyHostToDevice, st));
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, dtok, B, N, dx_, st);
        const unsigned gBN = grid1((int64_t) B * N);
        for (int l = 0; l < D.n_layers; ++l) {
            const DfLayer& y = D.L[(size_t) l];
            k_rms_ld<<<B, kThreads, 0, st>>>(dx_, N, y.attn_norm, dh_, N, N, D.eps);
            dmm(y.attn_conv, dh_, ddyn, B);
            k_df_conv<<<gBN, kThreads, 0, st>>>(dh_, ddyn, proj, y.attn_cbase, 0, dh2_, B, N, D.conv_g, D.conv_k);
            dmm(y.wq, dh2_, dq_, B);
            dmm(y.wk, dh2_, dk_, B);
            dmm(y.wv, dh2_, dv_, B);
            k_df_norm_rope<<<dim3(B, D.H), 128, 0, st>>>(dq_, N, y.q_norm, D.eps, pos, D.rope_base);
            k_df_norm_rope<<<dim3(B, D.HKV), 128, 0, st>>>(dk_, KV, y.k_norm, D.eps, pos, D.rope_base);
            k_df_attn<<<dim3(B, D.H), 128, 0, st>>>(dq_, dk_, dv_, df_kr[(size_t) l], df_vr[(size_t) l], y.sinks, D.H, D.HKV, B,
                                                     pos, df_R, D.window, scale, dat);
            dmm(y.wo, dat, dy_, B);
            k_df_conv<<<gBN, kThreads, 0, st>>>(dy_, ddyn, proj, y.attn_cbase + (size_t) D.conv_k * N, 1, dy2_, B, N, D.conv_g,
                                                D.conv_k);
            k_add_norm<<<B, kThreads, 0, st>>>(dx_, dy2_, nullptr, y.ffn_norm, dh_, N, D.eps, false);
            dmm(y.ffn_conv, dh_, ddyn, B);
            k_df_conv<<<gBN, kThreads, 0, st>>>(dh_, ddyn, proj, y.ffn_cbase, 0, dh2_, B, N, D.conv_g, D.conv_k);
            dmm(y.gate, dh2_, dg_, B);
            dmm(y.up, dh2_, du_, B);
            k_df_swiglu<<<grid1((int64_t) B * D.FF), kThreads, 0, st>>>(dg_, du_, (int64_t) B * D.FF);
            dmm(y.down, dg_, dy_, B);
            k_df_conv<<<gBN, kThreads, 0, st>>>(dy_, ddyn, proj, y.ffn_cbase + (size_t) D.conv_k * N, 1, dy2_, B, N, D.conv_g,
                                                D.conv_k);
            k_df_add<<<gBN, kThreads, 0, st>>>(dx_, dy2_, (int64_t) B * N);
        }
        k_rms_ld<<<B, kThreads, 0, st>>>(dx_, N, D.out_norm, dhout, N, N, D.eps);
        dmm(m->head, dhout + N, dlog, S);   // the block's rows 1 .. B-1 (row 0 is the anchor token)
        CK(cudaMemcpyAsync(h_dlog, dlog, (size_t) S * g.V * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        // the top-k candidates of each position (descending logit) and their features [lp, p, lp - lp_0, rank / (k-1), rank == 0]
        for (int s = 0; s < S; ++s) {
            const float* lg = h_dlog + (size_t) s * g.V;
            int ci[8];
            float cv[8];
            int nc = 0;
            float mx = -INFINITY;
            for (int v = 0; v < g.V; ++v) {
                const float a = lg[v];
                mx = std::max(mx, a);
                if (nc < tk || a > cv[nc - 1]) {
                    int p = nc < tk ? nc++ : tk - 1;
                    while (p > 0 && cv[p - 1] < a) { cv[p] = cv[p - 1]; ci[p] = ci[p - 1]; --p; }
                    cv[p] = a;
                    ci[p] = v;
                }
            }
            double sum = 0;
            for (int v = 0; v < g.V; ++v) sum += std::exp((double) (lg[v] - mx));
            const double lz = (double) mx + std::log(sum);
            for (int r = 0; r < tk; ++r) {
                const int nd = s * tk + r;
                const float lp = (float) ((double) cv[r] - lz), lp0 = (float) ((double) cv[0] - lz);
                h_cand[nd] = ci[r];
                float* f = h_feat + (size_t) nd * 5;
                f[0] = lp;
                f[1] = std::exp(lp);
                f[2] = lp - lp0;
                f[3] = tk == 1 ? 0.f : (float) r / (float) (tk - 1);
                f[4] = r == 0 ? 1.f : 0.f;
            }
        }
        CK(cudaMemcpyAsync(dcand, h_cand, (size_t) nodes * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(dfeat, h_feat, (size_t) nodes * 5 * 4, cudaMemcpyHostToDevice, st));
        // ---- LiLiCorr
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, dcand, nodes, N, lc_emb, st);
        dmm(D.token_proj, lc_emb, lc_tok, nodes);
        bias(lc_tok, D.token_b, hid, nodes);
        dmm(D.pass_proj, dhout + N, lc_pass, S);
        bias(lc_pass, D.pass_b, hid, S);
        k_lc_feat<<<nodes, kThreads, 0, st>>>(dfeat, D.fnorm_w, D.fnorm_b, D.f1_w, D.f1_b, lc_f0, hid);
        f32mm(D.f3_w, lc_f0, lc_f, hid, hid, nodes);
        bias(lc_f, D.f3_b, hid, nodes);
        const unsigned gNH = grid1((int64_t) nodes * hid);
        k_lc_combine<<<gNH, kThreads, 0, st>>>(lc_tok, lc_pass, lc_f, D.slot_e, D.rank_e, lc_x, hid, tk, nodes);
        dmm(D.ctx_proj, df_anchor, lc_anc, 1);
        bias(lc_anc, D.ctx_b, hid, 1);
        for (const LcLayer& y : D.LL) {
            k_rms_ld<<<nodes, kThreads, 0, st>>>(lc_x, hid, y.attn_norm, lc_a, hid, hid, D.eps);
            f32mm(y.in_proj, lc_a, lc_qkv, hid, 3 * hid, nodes);
            bias(lc_qkv, y.in_proj_b, 3 * hid, nodes);
            k_lc_attn<<<dim3(nodes, D.heads), 128, 0, st>>>(lc_qkv, D.abias, nodes, hid, 128, scale, lc_ao);
            dmm(y.out_proj, lc_ao, lc_t, nodes);
            bias(lc_t, y.out_b, hid, nodes);
            k_df_add<<<gNH, kThreads, 0, st>>>(lc_x, lc_t, (int64_t) nodes * hid);
            k_rms_ld<<<nodes, kThreads, 0, st>>>(lc_x, hid, y.mlp_norm, lc_a, hid, hid, D.eps);
            dmm(y.mlp0, lc_a, lc_m0, nodes);
            bias(lc_m0, y.mlp0_b, D.mlp, nodes, 1);
            dmm(y.mlp2, lc_m0, lc_t, nodes);
            bias(lc_t, y.mlp2_b, hid, nodes);
            k_df_add<<<gNH, kThreads, 0, st>>>(lc_x, lc_t, (int64_t) nodes * hid);
        }
        k_rms_ld<<<nodes, kThreads, 0, st>>>(lc_x, hid, D.out_norm_l, lc_a, hid, hid, D.eps);
        k_rms_ld<<<1, kThreads, 0, st>>>(lc_anc, hid, D.anchor_norm, lc_ancn, hid, hid, D.eps);
        k_lc_factor_in<<<gNH, kThreads, 0, st>>>(lc_a, lc_ancn, lc_fin, hid, nodes);
        f32mm(D.fin_w, lc_fin, lc_fac, 3 * hid, hid, nodes);
        bias(lc_fac, D.fin_b, hid, nodes, 1);
        f32mm(D.out_w, lc_fac, lc_out, hid, D.fdim, nodes);
        bias(lc_out, D.out_b, D.fdim, nodes);
        k_lc_l2<<<nodes, kThreads, 0, st>>>(lc_out, D.fdim, D.vec_eps);
        f32mm(D.in_w, lc_fac, lc_in, hid, D.fdim, nodes);
        bias(lc_in, D.in_b, D.fdim, nodes);
        k_lc_l2<<<nodes, kThreads, 0, st>>>(lc_in, D.fdim, D.vec_eps);
        f32mm(D.aout_w, lc_ancn, lc_aout, hid, D.fdim, 1);
        bias(lc_aout, D.aout_b, D.fdim, 1);
        k_lc_l2<<<1, kThreads, 0, st>>>(lc_aout, D.fdim, D.vec_eps);
        k_lc_scores<<<dim3(S, tk), kThreads, 0, st>>>(lc_in, lc_out, lc_aout, tk, D.fdim, D.logit_scale, lc_sc);
        const size_t nsc = (size_t) tk + (size_t) (S - 1) * tk * tk;
        CK(cudaMemcpyAsync(h_sc, lc_sc, nsc * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        // the greedy walk (common/speculative.cpp): slot 0 from the anchor, then from the predecessor's candidate
        drafts.clear();
        int pred = 0;
        for (int s = 0; s < std::min(n, S); ++s) {
            const float* sc = s == 0 ? h_sc : h_sc + tk + ((size_t) (s - 1) * tk + pred) * tk;
            pred = (int) (std::max_element(sc, sc + tk) - sc);
            drafts.push_back(h_cand[s * tk + pred]);
        }
    }
    /// One DFlash step (greedy): `x_tok` is the next token (decided, not processed).  Returns the accepted count a;
    /// picks[0 .. a] are the window's picks.
    int dflash_step(int x_tok, std::vector<int32_t>& drafts, int32_t* picks) {
        const int N = m->g.N;
        const auto t0 = Clock::now();
        df_draft(x_tok, df_n, drafts);
        const auto t1 = Clock::now();
        if (!warming) df_draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int D = (int) drafts.size(), T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        forward(T, pos, T, true);
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(df_anchor, df_g + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        const int64_t before = pos;
        pos += a + 1;
        if (!warming) {
            df_offered += D;
            df_accepted += a;
            ++df_steps;
            df_verify_ms += ms_since(t1);
        }
        if (before / adapt_every != pos / adapt_every) rebalance(adapt_swaps);
        return a;
    }

    const Layer& moe_layer(int mi) const {
        if (mi == m->n_moe) return m->mtp_moe;
        for (const Layer& L : m->layers)
            if (L.kind == Kind::Moe && L.idx == mi) return L;
        std::fprintf(stderr, "strata-nh: no MoE layer %d\n", mi);
        std::exit(1);
    }

    void init(const Model& model, int ctx_, int tmax_) {
        m = &model;
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        const Geo& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.N, Kp = (size_t) g.top_k;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); y2 = dalloc(T * N);
        hmain = dalloc(T * N); hlast = dalloc(N); seeds = dalloc(T * N); hmtp = dalloc(T * N);
        mtp_cat = dalloc(T * 2 * N);
        zx = dalloc(T * g.in_proj); xc = dalloc(T * g.conv_ch); yr = dalloc(T * g.ssm_inner); yg = dalloc(T * g.ssm_inner);
        qx = dalloc(T * g.H * kHD); kx = dalloc(T * g.HKV * kHD); vx = dalloc(T * g.HKV * kHD); att = dalloc(T * g.H * kHD);
        logits_r = dalloc(T * g.E); ewts = dalloc(T * Kp);
        hl = dalloc(T * g.LAT); lat = dalloc(T * g.LAT); eparts = dalloc(T * Kp * g.LAT); su = dalloc(T * g.FS);
        logits = dalloc((size_t) kVerify * g.V); logits_m = dalloc((size_t) g.V);
        ubuf = dalloc(T * g.FF);
        const int max_in = std::max({2 * g.N, g.ssm_inner, g.H * kHD, g.FS, g.LAT});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes(std::max(g.N, g.LAT), 1) * T));
        CK(cudaMalloc(&xq_l, K::native_q8_1_bytes(g.LAT, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        xg16 = dalloc_t<__half>(T * Kp * g.LAT);
        hb16 = dalloc_t<__half>(T * g.FF);
        q16 = dalloc_t<__half>(T * g.H * kHD);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-nh: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * g.E);
        {
            std::string gerr;
            int64_t big = (int64_t) g.FF * g.LAT;
            auto take = [&](const Mat& w) { big = std::max<int64_t>(big, (int64_t) w.n_in * w.n_out); };
            for (const Layer& L : m->layers) {
                for (const Mat* w : {&L.ssm_in, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.moe.lat_down, &L.moe.lat_up,
                                     &L.moe.sh_up, &L.moe.sh_down})
                    take(*w);
            }
            if (m->has_mtp) {
                take(m->eh_proj);
                for (const Mat* w : {&m->mtp_attn.wq, &m->mtp_attn.wo, &m->mtp_moe.moe.sh_up, &m->mtp_moe.moe.sh_down}) take(*w);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-nh: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        // the grouped kernels' scratch for a rows batch (at most hybrid_t rows: a bigger batch streams)
        const size_t rows_cap = (size_t) std::max(std::min(hybrid_t, tmax), kVerify);
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (rows_cap * Kp), (int64_t) g.FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.E) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.E) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        rinv = dalloc_t<int32_t>(T * Kp);
        rcnt = dalloc_t<int32_t>((size_t) g.E);
        rcur = dalloc_t<int32_t>((size_t) g.E);
        rbounds = dalloc_t<int32_t>((size_t) g.E + 1);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        tok_m = dalloc_t<int32_t>(T);
        pos0_m = dalloc_t<int32_t>(1);
        pick_m = dalloc_t<int32_t>(1);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.H * std::min<int64_t>((int64_t) T, 16));
        pacc = dalloc((size_t) att_rows * n_chunk * kHD);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        const int vcols = g.in_proj - g.ssm_inner;   // xBC | dt
        for (int i = 0; i < m->n_mamba; ++i) {
            conv.push_back(dalloc((size_t) g.conv_ch * (g.ssm_conv - 1)));
            S.push_back(dalloc((size_t) g.ssm_H * kSsmP * kSsmN));
            vbuf.push_back(dalloc((size_t) kVerify * vcols));
        }
        const size_t kv_cells = (size_t) g.HKV * ctx;
        const size_t kv_row = kv_fmt == 0 ? kHD * 2 : kv_fmt == 1 ? kHD + 8 : kHD / 2 + 8;   // bytes a cell-head
        for (int i = 0; i < m->n_attn + (m->has_mtp ? 1 : 0); ++i) {
            if (kv_fmt == 0) {
                kc.push_back(dalloc_t<__half>(kv_cells * kHD));
                vc.push_back(dalloc_t<__half>(kv_cells * kHD));
            } else {
                const size_t cb = kv_cells * (kv_fmt == 1 ? kHD : kHD / 2);
                kq.push_back(dalloc_t<uint8_t>(cb));
                vq.push_back(dalloc_t<uint8_t>(cb));
                ks.push_back(dalloc_t<__half>(kv_cells * 4));
                vs.push_back(dalloc_t<__half>(kv_cells * 4));
            }
        }
        if (kv_fmt != 0) {
            kv_tk = dalloc_t<__half>((size_t) std::min(ctx, kKvSlice) * kHD);
            kv_tv = dalloc_t<__half>((size_t) std::min(ctx, kKvSlice) * kHD);
        }
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t n_kv = (size_t) (m->n_attn + (m->has_mtp ? 1 : 0));
        std::fprintf(stderr, "strata-nh: context %d cells (%s KV, %.2f GiB), prompt chunk %d, %.2f GiB of VRAM free\n",
                     ctx, kv_fmt == 0 ? "FP16" : kv_fmt == 1 ? "Q8_0" : "Q4_0",
                     (double) n_kv * 2 * kv_cells * kv_row / (1 << 30), tmax, (double) fr / (1 << 30));
    }

    void reset() {
        const Geo& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            CK(cudaMemsetAsync(conv[l], 0, (size_t) g.conv_ch * (g.ssm_conv - 1) * 4, st));
            CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_H * kSsmP * kSsmN * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.N * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
    }

    // ---- matmuls: a decode batch on the native MMVQ (q8_1 activations), a prompt chunk on FP16 GEMMs
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void mmq(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        if (w.type == kTypeNvfp4) {   // --nvfp4 native: iq_kernels' NVFP4 MMVQ, then the tensor scale
            for (int c0 = 0; c0 < T; c0 += 8)
                K::iq_mmvq(kTypeNvfp4, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, std::min(8, T - c0), st);
            k_scale_f32<<<grid1((int64_t) T * w.n_out), kThreads, 0, st>>>(out, (int64_t) T * w.n_out, w.s2);
            return;
        }
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT && w.type == kTypeNvfp4) {   // NVFP4 -> FP16 in the GEMM's scratch (row slices), FP16 GEMM
            const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(w.n_out, gemm.scratch_elems() / w.n_in));
            const size_t rb = (size_t) w.n_in / 64 * 36;
            for (int64_t r0 = 0; r0 < w.n_out; r0 += rows) {
                const int64_t n = std::min<int64_t>(rows, w.n_out - r0);
                K::iq_dequant_f16(kTypeNvfp4, w.d + (size_t) r0 * rb, n * w.n_in, gemm.scratch(), st);
                gemm.f16((const uint16_t*) x16, gemm.scratch(), out + r0, T, n, w.n_in, w.n_out);
            }
            k_scale_f32<<<grid1((int64_t) T * w.n_out), kThreads, 0, st>>>(out, (int64_t) T * w.n_out, w.s2);
            return;
        }
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// out [T][n_out] = W [n_out][n_in] (f32) x: the router (FP32, as llama.cpp keeps it)
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const int tm = n_out > 64 ? 4 : 1;
        const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
        const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
        const int kz = (n_in / want + 15) / 16 * 16;
        const int splits = (n_in + kz - 1) / kz;
        float* dst = splits > 1 ? sk_part : out;
        if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        if (splits > 1) k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
    }

    // ---- the sublayers: each reads h (the normed input) and writes its output to y (MoE: y + y2)
    void mamba(const Layer& L, int T, bool verify) {
        const Geo& g = m->g;
        const int li = L.idx;
        mm(L.ssm_in, h, zx, T);   // [T][z 8192 | xBC 10240 | dt 128]
        const int vcols = g.in_proj - g.ssm_inner;
        if (verify)   // the window's xBC | dt rows, for the commit
            CK(cudaMemcpy2DAsync(vbuf[(size_t) li], (size_t) vcols * 4, zx + g.ssm_inner, (size_t) g.in_proj * 4,
                                 (size_t) vcols * 4, (size_t) T, cudaMemcpyDeviceToDevice, st));
        k_conv<<<grid1(g.conv_ch, 128), 128, 0, st>>>(conv[(size_t) li], zx + g.ssm_inner, g.in_proj, L.conv_w, L.conv_b, xc,
                                                      g.conv_ch, T, !verify);
        k_ssm<<<g.ssm_H, 256, 0, st>>>(S[(size_t) li], xc, g.conv_ch, zx + g.ssm_inner + g.conv_ch, g.in_proj, L.dt_b, L.A,
                                       L.D, yr, g.ssm_H, g.ssm_G, T, !verify);
        k_gate_norm<<<dim3(g.ssm_G, T), kThreads, 0, st>>>(yr, zx, g.in_proj, L.ssm_norm, yg, g.ssm_inner,
                                                           g.ssm_inner / g.ssm_G, g.eps);
        mm(L.ssm_out, yg, y, T);
    }

    /// attention()'s cache half over the quantized KV (kv_fmt 1 / 2): store, then the prompt's FP16 GEMMs over
    /// dequantized slices of kKvSlice cells (scores slice by slice, values accumulated with beta 1) or the decode's
    /// split-K kernels.  qx / kx / vx in, att out.
    void attention_q(int li, int T, int64_t p0, const int32_t* dpos, float scale) {
        const Geo& g = m->g;
        const int H = g.H, HKV = g.HKV;
        const bool q4 = kv_fmt == 2;
        uint8_t *Kq = kq[(size_t) li], *Vq = vq[(size_t) li];
        __half *Ks = ks[(size_t) li], *Vs = vs[(size_t) li];
        if (q4) k_kv_store_q<1><<<dim3(HKV, T), kHD, 0, st>>>(kx, vx, Kq, Ks, Vq, Vs, HKV, dpos, ctx);
        else k_kv_store_q<0><<<dim3(HKV, T), kHD, 0, st>>>(kx, vx, Kq, Ks, Vq, Vs, HKV, dpos, ctx);
        if (T > kGemmT) {
            const int P = (int) p0 + T;
            const int ldp = (P + 7) / 8 * 8;
            const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
            const int G = H / HKV;
            const float one = 1.0f, zero = 0.0f;
            auto deq = [&](const uint8_t* q, const __half* s, size_t cell0, int n, __half* out) {
                if (q4) k_kv_deq<1><<<n, kHD, 0, st>>>(q, s, cell0, out);
                else k_kv_deq<0><<<n, kHD, 0, st>>>(q, s, cell0, out);
            };
            k_to_f16<<<grid1((int64_t) T * H * kHD), kThreads, 0, st>>>(qx, q16, (int64_t) T * H * kHD);
            for (int t0 = 0; t0 < T; t0 += tq) {
                const int nt = std::min(tq, T - t0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * kHD;
                    float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                    for (int c0 = 0; c0 < P; c0 += kKvSlice) {
                        const int pc = std::min(kKvSlice, P - c0);
                        deq(Kq, Ks, (size_t) kvh * ctx + c0, pc, kv_tk);
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, pc, nt, kHD, &one, kv_tk, CUDA_R_16F,
                                                       kHD, 0, Qh, CUDA_R_16F, H * kHD, kHD, &zero, Sh + c0, CUDA_R_32F,
                                                       ldp, (long long) nt * ldp, G, CUBLAS_COMPUTE_32F,
                                                       CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-nh: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0, scale);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                    float* Oh = att + ((size_t) t0 * H + kvh * G) * kHD;
                    for (int c0 = 0; c0 < P; c0 += kKvSlice) {
                        const int pc = std::min(kKvSlice, P - c0);
                        deq(Vq, Vs, (size_t) kvh * ctx + c0, pc, kv_tv);
                        const float* beta = c0 == 0 ? &zero : &one;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, kHD, nt, pc, &one, kv_tv, CUDA_R_16F,
                                                       kHD, 0, Sh + c0, CUDA_R_16F, ldp, (long long) nt * ldp, beta, Oh,
                                                       CUDA_R_32F, H * kHD, kHD, G, CUBLAS_COMPUTE_32F,
                                                       CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-nh: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
            }
            return;
        }
        const int rows_per = att_rows / H;
        for (int t0 = 0; t0 < T; t0 += rows_per) {
            const int nt = std::min(rows_per, T - t0);
            const int nc = (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
            if (q4) k_attn_part_q<1><<<dim3(nc, nt * H), kThreads, 0, st>>>(qx, Kq, Ks, Vq, Vs, dpos, t0, H, HKV, ctx, scale,
                                                                           n_chunk, pacc, pm, pl);
            else k_attn_part_q<0><<<dim3(nc, nt * H), kThreads, 0, st>>>(qx, Kq, Ks, Vq, Vs, dpos, t0, H, HKV, ctx, scale,
                                                                        n_chunk, pacc, pm, pl);
            k_attn_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, dpos, t0, H, n_chunk, att);
        }
    }

    void attention(const Layer& L, int T, int64_t p0, const int32_t* dpos) {
        const Geo& g = m->g;
        const int H = g.H, HKV = g.HKV, li = L.idx;
        prep(h, g.N, T, xq);
        apply(L.wq, xq, qx, T);
        apply(L.wk, xq, kx, T);
        apply(L.wv, xq, vx, T);
        const float scale = 1.0f / sqrtf((float) kHD);
        if (kv_fmt != 0) {
            attention_q(li, T, p0, dpos, scale);
            mm(L.wo, att, y, T);
            return;
        }
        k_kv_store<<<dim3(HKV, T), kHD, 0, st>>>(kx, vx, kc[(size_t) li], vc[(size_t) li], HKV, dpos, ctx);
        if (T > kGemmT) {
            // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs per KV head over the FP16 cache, the causal
            // softmax in between; the queries in slices whose scores fit
            const int P = (int) p0 + T;
            const int ldp = (P + 7) / 8 * 8;
            const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
            const int G = H / HKV;
            const float one = 1.0f, zero = 0.0f;
            k_to_f16<<<grid1((int64_t) T * H * kHD), kThreads, 0, st>>>(qx, q16, (int64_t) T * H * kHD);
            for (int t0 = 0; t0 < T; t0 += tq) {
                const int nt = std::min(tq, T - t0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Kh = kc[(size_t) li] + (size_t) kvh * ctx * kHD;
                    const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * kHD;
                    float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, kHD, &one, Kh, CUDA_R_16F, kHD, 0,
                                                   Qh, CUDA_R_16F, H * kHD, kHD, &zero, Sh, CUDA_R_32F, ldp,
                                                   (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-nh: attention scores GEMM failed\n");
                        std::exit(1);
                    }
                }
                k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0, scale);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Vh = vc[(size_t) li] + (size_t) kvh * ctx * kHD;
                    const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                    float* Oh = att + ((size_t) t0 * H + kvh * G) * kHD;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, kHD, nt, P, &one, Vh, CUDA_R_16F, kHD, 0,
                                                   Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                   H * kHD, kHD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-nh: attention values GEMM failed\n");
                        std::exit(1);
                    }
                }
            }
        } else {
            const int rows_per = att_rows / H;
            for (int t0 = 0; t0 < T; t0 += rows_per) {
                const int nt = std::min(rows_per, T - t0);
                const int nc = (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                k_attn_part<<<dim3(nc, nt * H), kThreads, 0, st>>>(qx, kc[(size_t) li], vc[(size_t) li], dpos, t0, H, HKV,
                                                                  ctx, scale, n_chunk, pacc, pm, pl);
                k_attn_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, dpos, t0, H, n_chunk, att);
            }
        }
        mm(L.wo, att, y, T);
    }

    /// The MoE block for T rows of h: y = latent_up(sum_k w_k expert_k(latent_down h)), y2 = the shared expert.
    void moe(const Layer& Lr, int T) {
        const Geo& g = m->g;
        const Moe& M = Lr.moe;
        const int mi = Lr.idx, N = g.N, Kp = g.top_k, LAT = g.LAT;   // FF / E: this layer's (M.layer_ff / layer_e)
        const int n_ent = T * Kp;
        f32mm(M.router, h, logits_r, N, M.layer_e, T);
        k_route<<<T, 512, 0, st>>>(logits_r, M.bias, M.layer_e, Kp, g.w_scale, g.w_norm, ids, ewts);
        prep(h, N, T, xq);                     // (h's image serves the latent and the shared expert's up)
        apply(M.lat_down, xq, hl, T);
        const int32_t* prow = nullptr;
        if (T <= hybrid_t) {
            // ---- rows: the resident experts on the GPU, the others on the CPU pool, side by side
            Hy& Hh = hy[(size_t) mi];
            quant(hl, LAT, T, xq_l);
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            const bool skipping = skip_thr > 0.f || skip_rel > 0.f;
            if (skipping) CK(cudaMemcpyAsync(h_w, ewts, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_hl, hl, (size_t) T * LAT * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, g.E, Hh.slot_of_d, M.slots, M.stride, grp_ptr, grp_start,
                                               n_groups, ent_dst, ent_tok);
            K::native_expert_grouped_relu2(M.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                           std::min<int64_t>(n_ent, g.E), n_ent, xq_l, escr, eparts, st);
            shared(M, T);
            const auto tw = Clock::now();
            CK(cudaEventSynchronize(ev_route));
            t_route_wait_ms += ms_since(tw);
            const auto t0 = Clock::now();
            // every entry: resident (0), prefetched (1), the PCIe share (2), the CPU's (3) or skipped (4)
            const int pb = (pf_cap > 0 && pf[mi & 1].mi == mi) ? (mi & 1) : -1;
            int n_skip = 0;
            std::vector<int> cls((size_t) n_ent, 0), miss_x;
            std::vector<int> pc_slot;   // expert -> its place in pcbuf (lazily sized)
            for (int e = 0; e < n_ent; ++e) {
                const int xid = h_ids[e];
                if (!warming) { Hh.score[(size_t) xid] += 1.0; Hh.total[(size_t) xid] += 1.0; }
                if (Hh.slot_of[(size_t) xid] >= 0) cls[(size_t) e] = 0;
                else if (pb >= 0 && pf[pb].slot[(size_t) xid] >= 0) cls[(size_t) e] = 1;
                else {
                    if (skipping) {
                        const float *wr = h_w + (size_t) (e / Kp) * Kp;
                        float wmax = 0.f;
                        for (int k = 0; k < Kp; ++k) wmax = std::max(wmax, wr[k]);
                        if (!warming) ++skip_tot;
                        if (h_w[e] < skip_thr || h_w[e] < skip_rel * wmax) {   // not computed: the row is zero
                            cls[(size_t) e] = 4;
                            if (!warming) ++skip_n;
                            continue;
                        }
                    }
                    cls[(size_t) e] = 3;
                    miss_x.push_back(xid);
                }
            }
            n_skip = 0;
            for (int e = 0; e < n_ent; ++e) n_skip += cls[(size_t) e] == 4;
            if (skip_renorm && n_skip > 0) {
                // each token row's kept weights scaled to sum to what the row summed to (k_combine reads ewts on the device)
                for (int t = 0; t < T; ++t) {
                    float all = 0.f, kept = 0.f;
                    for (int k = 0; k < Kp; ++k) {
                        const int e = t * Kp + k;
                        all += h_w[e];
                        h_w2[e] = cls[(size_t) e] == 4 ? 0.f : h_w[e];
                        kept += h_w2[e];
                    }
                    const float sc = kept > 0.f ? all / kept : 1.f;
                    for (int k = 0; k < Kp; ++k) h_w2[t * Kp + k] *= sc;
                }
                CK(cudaMemcpyAsync(ewts, h_w2, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
            }
            // the PCIe share: that part of the distinct missed experts, the ones with the most rows first (in pick order
            // on ties: the router's ranking), uploaded on the copy stream while the CPU computes the others
            int n_pc = 0;
            if (pcie_cap > 0 && pcie_frac > 0.f && !miss_x.empty()) {
                std::vector<int> cnt_x;
                std::vector<int> distinct;
                std::map<int, int> c;
                for (int x : miss_x) if (c[x]++ == 0) distinct.push_back(x);
                std::stable_sort(distinct.begin(), distinct.end(), [&](int a, int b) { return c[a] > c[b]; });
                const int want = std::min<int>(pcie_cap, (int) std::lround(pcie_frac * (double) distinct.size()));
                pc_slot.assign((size_t) g.E, -1);
                CK(cudaStreamWaitEvent(cst2, ev_trfree, 0));
                for (int i = 0; i < want; ++i) {
                    const int x = distinct[(size_t) i];
                    pc_slot[(size_t) x] = n_pc;
                    CK(cudaMemcpyAsync(pcbuf + (size_t) n_pc * tr_stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice, cst2));
                    CK(cudaMemcpyAsync(pcbuf + (size_t) n_pc * tr_stride + M.L.down_off, M.down_of(x), M.d_bytes,
                                       cudaMemcpyHostToDevice, cst2));
                    ++n_pc;
                }
                CK(cudaEventRecord(ev_pc, cst2));
                for (int e = 0; e < n_ent; ++e)
                    if (cls[(size_t) e] == 3 && pc_slot[(size_t) h_ids[e]] >= 0) cls[(size_t) e] = 2;
            }
            // the transient experts' groups (prefetched + PCIe), built here, one grouped call after their uploads
            {
                std::map<int, std::vector<int>> grp;   // expert -> its entries (ascending id, entry order)
                for (int e = 0; e < n_ent; ++e)
                    if (cls[(size_t) e] == 1 || cls[(size_t) e] == 2) grp[h_ids[e]].push_back(e);
                if (!grp.empty()) {
                    int ng = 0, pos = 0;
                    for (const auto& [x, es] : grp) {
                        const bool pre = cls[(size_t) es[0]] == 1;
                        h_tgp[ng] = (unsigned long long) (pre ? pfbuf[pb] + (size_t) pf[pb].slot[(size_t) x] * tr_stride
                                                              : pcbuf + (size_t) pc_slot[(size_t) x] * tr_stride);
                        h_tgs[ng] = pos;
                        for (int e : es) {
                            h_ted[pos] = e;
                            h_tet[pos] = e / Kp;
                            ++pos;
                        }
                        ++ng;
                    }
                    h_tgs[ng] = pos;
                    *h_tng = ng;
                    CK(cudaMemcpyAsync(d_tgp, h_tgp, (size_t) ng * 8, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tgs, h_tgs, (size_t) (ng + 1) * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_ted, h_ted, (size_t) pos * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tet, h_tet, (size_t) pos * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_tng, h_tng, 4, cudaMemcpyHostToDevice, st));
                    if (n_pc > 0) CK(cudaStreamWaitEvent(st, ev_pc, 0));
                    if (pb >= 0) CK(cudaStreamWaitEvent(st, ev_pf[pb], 0));
                    K::native_expert_grouped_relu2(M.L, d_tgp, d_tgs, d_tng, d_ted, d_tet, ng, pos, xq_l, escr, eparts, st);
                }
                CK(cudaEventRecord(ev_trfree, st));
                if (pb >= 0) {
                    CK(cudaEventRecord(ev_pffree[pb], st));
                    pf[pb].mi = -1;
                }
            }
            jobs.clear();
            int n_miss = 0, n_res = 0, n_pf = 0;
            for (int e = 0; e < n_ent; ++e) {
                const int c = cls[(size_t) e];
                h_miss[e] = c >= 3;   // (a skipped entry merges a zero row)
                if (c == 4) std::memset(cpu_out + (size_t) e * LAT, 0, (size_t) LAT * 4);
                n_res += c == 0;
                n_pf += c == 1;
                if (c != 3) continue;
                const int xid = h_ids[e];
                ++n_miss;
                strata::nemotron_h::CpuJob j;
                j.up = M.up_of(xid);
                j.down = M.down_of(xid);
                j.tok = e / Kp;
                j.out = cpu_out + (size_t) e * LAT;
                jobs.push_back(j);
            }
            if (!warming) {
                hy_cpu += n_miss;
                hy_gpu += n_res;
                tr_pf_hit += n_pf;
                tr_pcie += n_ent - n_miss - n_res - n_pf - n_skip;
            }
            if (n_miss > 0 || n_skip > 0) {
                if (n_miss > 0) pool->run(M.fmt, h_hl, T, jobs.data(), (int) jobs.size());
                hy_cpu_ms += ms_since(t0);
                CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * LAT * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
                k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, LAT);
            }
        } else {
            // ---- a prompt chunk: the entries sorted by expert, every expert's rows as two FP16 GEMMs, its weights from
            // its slot or from the stream
            CK(cudaMemsetAsync(rcnt, 0, (size_t) g.E * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, g.E, rbounds, rcur);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            k_gather_f16<<<grid1((int64_t) n_ent * LAT), kThreads, 0, st>>>(hl, ent_tok, xg16, n_ent, LAT);
            CK(cudaMemcpyAsync(h_cnt, rcnt, (size_t) g.E * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            Hy& Hh = hy[(size_t) mi];
            int b0 = 0;
            for (int xid = 0; xid < M.layer_e; ++xid) {
                const int c = h_cnt[xid];
                if (!warming) { Hh.score[(size_t) xid] += c; Hh.total[(size_t) xid] += c; }
                const bool res = Hh.slot_of[(size_t) xid] >= 0;
                long item = -1;
                const uint8_t* w = nullptr;
                if (res) w = M.slots + (size_t) Hh.slot_of[(size_t) xid] * M.stride;
                else {
                    item = xs_pos++;
                    w = xs.take(item, st);
                    ++hy_streamed;
                }
                if (c > 0) {
                    gemm.native((const uint16_t*) (xg16 + (size_t) b0 * LAT), M.L.gu_type, w, ubuf, c, M.layer_ff, LAT);
                    k_relu2_f16<<<grid1((int64_t) c * M.layer_ff), kThreads, 0, st>>>(ubuf, hb16, (int64_t) c * M.layer_ff);
                    gemm.native((const uint16_t*) hb16, M.L.d_type, w + M.L.down_off, eparts + (size_t) b0 * LAT, c, LAT, M.layer_ff);
                }
                if (!res) xs.done(item, st);
                b0 += c;
            }
            prow = rinv;
            shared(M, T);
        }
        k_combine<<<T, kThreads, 0, st>>>(eparts, prow, ewts, Kp, lat, LAT);
        mm(M.lat_up, lat, y, T);
    }
    /// y2 = down(relu(up h)^2), h's image already in xq / x16 (prep in moe)
    void shared(const Moe& M, int T) {
        const Geo& g = m->g;
        prep(h, g.N, T, xq);
        apply(M.sh_up, xq, su, T);
        k_relu2<<<grid1((int64_t) T * g.FS), kThreads, 0, st>>>(su, (int64_t) T * g.FS);
        mm(M.sh_down, su, y2, T);
    }

    /// The prompt-path stream plan for a forward over the MoE slot indices `mis` (in order).
    void stream_plan(const std::vector<int>& mis) {
        std::vector<ExpertStream::Item> items;
        for (int mi : mis) {
            const Moe& M = moe_layer(mi).moe;
            const Hy& Hh = hy[(size_t) mi];
            for (int xid = 0; xid < M.layer_e; ++xid)
                if (Hh.slot_of[(size_t) xid] < 0)
                    items.push_back({M.up_of(xid), M.down_of(xid), M.up_bytes, M.d_bytes});
        }
        xs_pos = 0;
        xs.begin(std::move(items));
    }

    void embed_inject(float* dst, int T, int64_t p0) {
        if (inject.empty()) return;
        const int N = m->g.N;
        for (auto it = inject.lower_bound(p0); it != inject.end() && it->first < p0 + T; ++it)
            CK(cudaMemcpyAsync(dst + (size_t) (it->first - p0) * N, it->second, (size_t) N * 4, cudaMemcpyHostToDevice, st));
    }

    /// The main model over T rows (device `tok`, positions p0 .., also *pos0): hmain = every row's final normed
    /// hidden; the last `heads` rows get logits and greedy picks.  `verify`: the Mamba state is left for commit().
    void forward(int T, int64_t p0, int heads, bool verify) {
        const Geo& g = m->g;
        const int N = g.N;
        if (T > hybrid_t) {
            std::vector<int> mis;
            for (int i = 0; i < m->n_moe; ++i) mis.push_back(i);
            stream_plan(mis);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        embed_inject(x, T, p0);
        k_rms_ld<<<T, kThreads, 0, st>>>(x, N, m->layers[0].norm, h, N, N, g.eps);
        const size_t nl = m->layers.size();
        if (dump_f) dump_buf.assign(nl * (size_t) T * N, 0.f);
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            const bool last = l + 1 == m->layers.size();
            const float* wn = last ? m->out_norm : m->layers[l + 1].norm;
            float* hout = last ? hmain : h;
            // the prefetch: the next sublayer is a MoE block in the rows path -> guess its experts now, upload them while
            // this sublayer runs
            if (df) df_tap_layer(l, T);   // the drafter's context: the residual at this sublayer's input
            const bool pfn = pf_cap > 0 && T <= hybrid_t && !last && L.kind != Kind::Moe &&
                             m->layers[l + 1].kind == Kind::Moe;
            if (pfn) prefetch_begin(m->layers[l + 1], T);
            if (L.kind == Kind::Mamba) {
                mamba(L, T, verify);
                k_add_norm<<<T, kThreads, 0, st>>>(x, y, nullptr, wn, hout, N, g.eps, false);
                if (pfn) prefetch_issue(m->layers[l + 1], T);
            } else if (L.kind == Kind::Attn) {
                attention(L, T, p0, pos0);
                k_add_norm<<<T, kThreads, 0, st>>>(x, y, nullptr, wn, hout, N, g.eps, false);
                if (pfn) prefetch_issue(m->layers[l + 1], T);
            } else {
                moe(L, T);
                k_add_norm<<<T, kThreads, 0, st>>>(x, y, y2, wn, hout, N, g.eps, false);
            }
            if (dump_f)
                CK(cudaMemcpyAsync(dump_buf.data() + l * (size_t) T * N, x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        }
        if (dump_f && !verify) {
            CK(cudaStreamSynchronize(st));
            for (int t = 0; t < T; ++t)
                for (size_t l = 0; l < nl; ++l) std::fwrite(dump_buf.data() + (l * (size_t) T + t) * N, 4, (size_t) N, dump_f);
        }
        if (T > hybrid_t) xs.finish();
        if (df) df_context(T, p0);
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, m->g.V, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t and the seed hidden sd[t]; hmtp gets
    /// each row's head-normed output.  `draft`: logits of the last row and its pick (pick_m).
    void forward_mtp(int T, int64_t p0, const float* sd, bool draft) {
        const Geo& g = m->g;
        const int N = g.N;
        if (T > hybrid_t) stream_plan({m->n_moe});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        embed_inject(y, T, p0);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, g.eps);        // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(sd, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, g.eps);   // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms_ld<<<T, kThreads, 0, st>>>(x, N, m->mtp_attn.norm, h, N, N, g.eps);
        attention(m->mtp_attn, T, p0, pos0_m);
        k_add_norm<<<T, kThreads, 0, st>>>(x, y, nullptr, m->mtp_post_norm, h, N, g.eps, false);
        moe(m->mtp_moe, T);
        k_add_norm<<<T, kThreads, 0, st>>>(x, y, y2, m->mtp_shnorm, hmtp, N, g.eps, true);
        if (T > hybrid_t) xs.finish();
        if (draft) head_rows(hmtp + (size_t) (T - 1) * N, 1, logits_m, pick_m);
    }

    /// After a verify window: the Mamba state takes the first n_keep rows (the conv and the scan again over the
    /// window's kept inputs).
    void commit(int n_keep) {
        const Geo& g = m->g;
        const int vcols = g.in_proj - g.ssm_inner;
        for (const Layer& L : m->layers) {
            if (L.kind != Kind::Mamba) continue;
            const float* vb = vbuf[(size_t) L.idx];
            k_conv<<<grid1(g.conv_ch, 128), 128, 0, st>>>(conv[(size_t) L.idx], vb, vcols, L.conv_w, L.conv_b, xc, g.conv_ch,
                                                          n_keep, true);
            k_ssm<<<g.ssm_H, 256, 0, st>>>(S[(size_t) L.idx], xc, g.conv_ch, vb + g.conv_ch, vcols, L.dt_b, L.A, L.D,
                                           nullptr, g.ssm_H, g.ssm_G, n_keep, true);
        }
    }

    // ---- the hybrid mode's slots
    /// The slots to the experts the routing used most: at most max_swaps replacements over all layers, between
    /// forwards; the scores halve each time.
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_NH_FIXED_SLOTS") != nullptr;
        if (off || warming) return;
        int left = max_swaps;
        CK(cudaStreamSynchronize(st));
        for (size_t mi = 0; mi < hy.size() && left > 0; ++mi) {
            Hy& Hh = hy[mi];
            const Moe& M = moe_layer((int) mi).moe;
            const int S_ = M.n_slots, ne = M.layer_e;   // this layer's experts only
            if (S_ <= 0 || S_ >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x_ = 0; x_ < ne; ++x_) order[(size_t) x_] = x_;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return Hh.score[(size_t) a] > Hh.score[(size_t) b]; });
            std::vector<int> victims;
            for (int s = 0; s < S_; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return Hh.score[(size_t) Hh.exp_in_slot[(size_t) a]] < Hh.score[(size_t) Hh.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S_ && left > 0 && vi < victims.size(); ++i) {
                const int xid = order[(size_t) i];
                if (Hh.slot_of[(size_t) xid] >= 0) continue;
                const int s = victims[vi];
                const int yid = Hh.exp_in_slot[(size_t) s];
                if (Hh.score[(size_t) yid] >= Hh.score[(size_t) xid]) break;
                ++vi;
                Hh.slot_of[(size_t) yid] = -1;
                Hh.slot_of[(size_t) xid] = s;
                Hh.exp_in_slot[(size_t) s] = xid;
                uint8_t* dst = M.slots + (size_t) s * M.stride;
                CK(cudaMemcpyAsync(dst, M.up_of(xid), M.up_bytes, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(dst + M.L.down_off, M.down_of(xid), M.d_bytes, cudaMemcpyHostToDevice, st));
                --left;
                ++hy_swaps;
                changed = true;
            }
            if (changed) CK(cudaMemcpyAsync(Hh.slot_of_d, Hh.slot_of.data(), Hh.slot_of.size() * 4, cudaMemcpyHostToDevice, st));
            for (double& v : Hh.score) v *= 0.5;
            CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable
        }
    }

    /// The tokens at the current position: a prompt in chunks, one token as a decode step.  The last token's logits
    /// stay in `logits` (its pick in `pick`), hlast its normed hidden.  With drafts on, the prompt also runs through
    /// the MTP block (its KV for every position: row q reads h_{q-1} and token q).
    template <class F> void process(const int32_t* toks, int64_t n, F on_chunk) {
        const int N = m->g.N;
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            forward(T, pos, i + T == n ? 1 : 0, false);
            if (m->has_mtp && mtp_d > 0 && prompt_mtp) {
                CK(cudaMemcpyAsync(seeds, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1) CK(cudaMemcpyAsync(seeds + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, seeds, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            if (df) CK(cudaMemcpyAsync(df_anchor, df_g + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            on_chunk(i + T);
            if (T == 1) {
                if (pos % adapt_every == 0) rebalance(adapt_swaps);
            }
        }
        if (n > 1) rebalance(swaps_after_prompt);
    }

    /// One speculative step with D MTP drafts (greedy), src/qwen35's spec_step: `x_tok` is the next token (decided,
    /// not processed), `fix` the drafts the last window accepted (the MTP block redoes their rows first).  Returns
    /// the accepted count a; picks[0 .. a] are the window's picks.
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks) {
        const int N = m->g.N;
        const int T0 = (int) fix.size() + 1;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        if (!fix.empty()) CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;
        for (int d = 1; d < D; ++d) {
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            forward_mtp(1, pos + d, seeds, true);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
        }
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        forward(T, pos, T, true);
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        const int64_t before = pos;
        pos += a + 1;
        if (before / adapt_every != pos / adapt_every) rebalance(adapt_swaps);
        return a;
    }

    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        // the rows path only (a chunk above --hybrid-rows would stream every non-resident expert: ~70 GB)
        const int32_t filler = 1010;
        std::vector<int32_t> toks((size_t) std::min(hybrid_t, kVerify), filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        if (df) {
            std::vector<int32_t> drafts;
            int32_t picks[16];
            dflash_step(filler, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& Hh : hy) std::fill(Hh.score.begin(), Hh.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = hy_streamed = 0;
        skip_n = skip_tot = 0;
        tr_pcie = tr_pf_sent = tr_pf_hit = 0;
        hy_cpu_ms = t_route_wait_ms = 0;
        reset();
        std::fprintf(stderr, "strata-nh: warmed up in %.1f s\n", ms_since(t0) / 1000.0);
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.V);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// MemAvailable from /proc/meminfo in bytes (0 when unknown).
size_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    size_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v * 1024;
    return 0;
}

/// The expert profile (Flash-Next's expert-profile.bin idea for this family): per MoE slot index (the main layers,
/// then the MTP block) every expert's routing count.  'NHP1', n_moe, n_expert, then f32 [n_moe][n_expert].  The slots
/// start with each layer's most-routed experts instead of 0 .. S - 1.
bool load_profile(const std::string& path, int n_moe, int E, std::vector<std::vector<double>>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hd[3];
    f.read((char*) hd, sizeof hd);
    if (!f || hd[0] != 0x3150484e || hd[1] != n_moe || hd[2] != E) return false;
    out.assign((size_t) n_moe, std::vector<double>((size_t) E, 0.0));
    std::vector<float> row((size_t) E);
    for (int i = 0; i < n_moe; ++i) {
        f.read((char*) row.data(), (std::streamsize) (E * 4));
        if (!f) return false;
        for (int x = 0; x < E; ++x) out[(size_t) i][(size_t) x] = row[(size_t) x];
    }
    return true;
}
bool save_profile(const std::string& path, const std::vector<Engine::Hy>& hy, int E) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;
        const int32_t hd[3] = {0x3150484e, (int32_t) hy.size(), E};
        f.write((const char*) hd, sizeof hd);
        std::vector<float> row((size_t) E);
        for (const Engine::Hy& H : hy) {
            for (int x = 0; x < E; ++x) row[(size_t) x] = (float) H.total[(size_t) x];
            f.write((const char*) row.data(), (std::streamsize) (E * 4));
        }
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

/// Every MoE layer (and the MTP block's) gets the same number of VRAM slots from the VRAM left after the engine's
/// buffers, the stream ring and the transient buffers (the PCIe share and the prefetch); `vram_gib` >= 0: that budget
/// instead, `slots` > 0: that many.  With `arena` (and the RAM for it) every expert is copied once into pinned host
/// memory, as Flash-Next's expert arena: the CPU pool reads it, and every upload (slots, the PCIe share, the prefetch,
/// the prompt stream) is a DMA at full PCIe speed.  `profile`: the slots start with its most-routed experts.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int slots, int ring, bool arena, int reserve_mib,
                   const std::string& profile, std::string& err) {
    const auto t0 = Clock::now();
    const Geo& g = m.g;
    std::vector<Moe*> all;
    for (Layer& L : m.layers)
        if (L.kind == Kind::Moe) all.push_back(&L.moe);
    if (m.has_mtp) all.push_back(&m.mtp_moe.moe);
    size_t max_stride = 0, total = 0;
    for (Moe* M : all) {
        max_stride = std::max(max_stride, M->stride);
        total += (size_t) M->layer_e * M->stride;   // each layer's own expert count (Bolt: 240-496)
    }
    // ---- the pinned arena
    bool have_arena = false;
    if (arena) {
        const size_t avail = mem_available();
        const size_t keep = (size_t) 6 << 30;   // the OS, the server, the vision encoder
        if (avail > total + keep) {
            uint8_t* base = nullptr;
            const char* how = "cudaHostAlloc";
#if !defined(_WIN32)
            // 2 MiB pages (transparent huge pages, madvise): a decode step reads ~700 experts of 3.8 MB each at random
            // places in 73 GB - with 4 KiB pages that is ~650,000 page walks per token; then registered with CUDA.
            // STRATA_NH_ARENA_4K=1: cudaHostAlloc's 4 KiB pages instead.
            if (std::getenv("STRATA_NH_ARENA_4K") == nullptr) {
                const size_t hp = (size_t) 2 << 20, len = (total + hp - 1) / hp * hp;
                void* q = mmap(nullptr, len + hp, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (q != MAP_FAILED) {
                    uint8_t* al = (uint8_t*) (((uintptr_t) q + hp - 1) & ~(uintptr_t) (hp - 1));
                    madvise(al, len, MADV_HUGEPAGE);
                    if (cudaHostRegister(al, len, cudaHostRegisterPortable) == cudaSuccess) {
                        base = al;
                        how = "huge pages, registered";
                    } else {
                        (void) cudaGetLastError();
                        munmap(q, len + hp);
                    }
                }
            }
#endif
            if (base == nullptr && cudaHostAlloc((void**) &base, total, cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                base = nullptr;
            }
            if (base != nullptr) {
                size_t off = 0;
                std::vector<int> first;   // the item index of each layer's expert 0
                int n_items = 0;
                for (Moe* M : all) {
                    M->host = base + off;
                    off += (size_t) M->layer_e * M->stride;
                    first.push_back(n_items);
                    n_items += M->layer_e;
                }
                // filled by 6 threads (from the page cache, or the disk on a cold start)
                std::atomic<int> next{0};
                std::vector<std::thread> ts;
                for (int k = 0; k < 6; ++k)
                    ts.emplace_back([&] {
                        for (int i; (i = next.fetch_add(1)) < n_items;) {
                            const size_t li = (size_t) (std::upper_bound(first.begin(), first.end(), i) - first.begin() - 1);
                            Moe& M = *all[li];
                            const int e = i - first[li];
                            uint8_t* dst = M.host + (size_t) e * M.stride;
                            std::memcpy(dst, M.src_up + (size_t) e * M.up_bytes, M.up_bytes);
                            std::memcpy(dst + M.L.down_off, M.src_down + (size_t) e * M.d_bytes, M.d_bytes);
                        }
                    });
                for (auto& t : ts) t.join();
                have_arena = true;
            }
            std::fprintf(stderr, "strata-nh: expert arena: %s (%s, %.2f GiB pinned, %.1f s)\n",
                         have_arena ? "every routed expert in pinned RAM" : "pinning failed, the mapped file is read", how,
                         have_arena ? (double) total / (1 << 30) : 0.0, ms_since(t0) / 1000.0);
        } else {
            std::fprintf(stderr, "strata-nh: expert arena: not enough RAM (%.1f GiB available, %.1f needed), the mapped file "
                         "is read\n", (double) avail / (1 << 30), (double) (total + keep) / (1 << 30));
        }
    }
    E.xs.dev = 0;
    CK(cudaGetDevice(&E.xs.dev));
    E.xs.init(std::max(4, ring), max_stride, have_arena);
    // ---- the transient buffers: the PCIe share's experts and two prefetch sets
    E.tr_stride = max_stride;
    if (E.pcie_frac <= 0.f) E.pcie_cap = 0;   // the share off: no buffer (its VRAM goes to the slots)
    if (E.pcie_cap > 0) E.pcbuf = dalloc_t<uint8_t>((size_t) E.pcie_cap * max_stride + kTail);
    if (E.pf_cap > 0)
        for (int b = 0; b < 2; ++b) E.pfbuf[b] = dalloc_t<uint8_t>((size_t) E.pf_cap * max_stride + kTail);
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) std::max(256, reserve_mib) << 20;   // --vram-reserve-mib (the server's: 700)
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30)) : (fr > reserve ? fr - reserve : 0);
    // the same slot count S for every layer, capped at the layer's own experts (a 240-expert layer never takes more
    // than 240): the largest S whose capped sum fits the budget
    auto cost = [&](int s) {
        size_t c = 0;
        for (Moe* M : all) c += (size_t) std::min(s, M->layer_e) * M->stride;
        return c;
    };
    int S_ = 0;
    if (slots > 0) S_ = std::min(g.E, slots);
    else {
        int lo = 0, hi = g.E;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (cost(mid) <= budget) lo = mid; else hi = mid - 1;
        }
        S_ = lo;
    }
    if (S_ < 0) S_ = 0;
    size_t used = 0;
    std::vector<std::vector<double>> prof;
    const bool have_prof = !profile.empty() && load_profile(profile, (int) all.size(), g.E, prof);
    if (!profile.empty())
        std::fprintf(stderr, "strata-nh: expert profile %s: %s\n", profile.c_str(), have_prof ? "loaded" : "not usable, slots start at 0 .. S-1");

    // ---- profile-mass slot allocation (STRATA_NH_MASS_SLOTS=1)
    // Start with the profile's hottest experts globally, normalized by each layer's slot size. This spends
    // bytes where they remove the most predicted CPU work instead of assuming a uniform slot count is best.
    // Falls back to uniform S_ unless explicitly enabled with a usable profile.
    const bool mass_slots = have_prof && std::getenv("STRATA_NH_MASS_SLOTS") != nullptr &&
                            std::string(std::getenv("STRATA_NH_MASS_SLOTS")) != "0";
    std::vector<int> layer_slots(all.size(), S_);
    if (mass_slots) {
        std::vector<strata::nemotron_h::ProfiledLayer> plan_layers;
        plan_layers.reserve(all.size());
        for (size_t mi = 0; mi < all.size(); ++mi)
            plan_layers.push_back({all[mi]->stride, all[mi]->layer_e, &prof[mi]});
        layer_slots = strata::nemotron_h::profile_slot_plan(plan_layers, budget);
        size_t assigned = 0;
        for (size_t mi = 0; mi < all.size(); ++mi) assigned += (size_t) layer_slots[mi] * all[mi]->stride;
        std::fprintf(stderr, "strata-nh: profile-mass slot allocation: total budget %zu MiB, assigned %zu MiB\n",
                     budget >> 20, assigned >> 20);
    }

    E.hy.resize(all.size());
    for (size_t mi = 0; mi < all.size(); ++mi) {
        Moe& M = *all[mi];
        const int Sl = mass_slots ? layer_slots[mi] : std::min(S_, M.layer_e);
        M.n_slots = Sl;
        used += (size_t) Sl * M.stride;
        CK(cudaMalloc(&M.slots, M.stride * (size_t) std::max(Sl, 1) + kTail));
        CK(cudaMemset(M.slots, 0, M.stride * (size_t) std::max(Sl, 1) + kTail));
        std::vector<int> order((size_t) M.layer_e);   // only this layer's experts (the tables stay g.E wide)
        for (int x = 0; x < M.layer_e; ++x) order[(size_t) x] = x;
        if (have_prof)
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return prof[mi][(size_t) a] > prof[mi][(size_t) b]; });
        Engine::Hy& H = E.hy[mi];
        H.slot_of.assign((size_t) g.E, -1);
        H.exp_in_slot.assign((size_t) Sl, 0);
        for (int s = 0; s < Sl; ++s) {
            const int x = order[(size_t) s];
            H.slot_of[(size_t) x] = s;
            H.exp_in_slot[(size_t) s] = x;
            CK(cudaMemcpy(M.slots + (size_t) s * M.stride, M.up_of(x), M.up_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(M.slots + (size_t) s * M.stride + M.L.down_off, M.down_of(x), M.d_bytes, cudaMemcpyHostToDevice));
        }
        // the profile is also the adaptive tier's prior: its counts scaled to `prior` tokens' worth of routing, so a
        // request's first counts move only the experts it really prefers (they halve with every rebalance as usual).
        // STRATA_NH_PRIOR_TOKENS: that weight (default 1024).
        H.score.assign((size_t) g.E, 0.0);
        if (have_prof) {
            static const double prior = std::getenv("STRATA_NH_PRIOR_TOKENS") ? std::atof(std::getenv("STRATA_NH_PRIOR_TOKENS")) : 1024.0;
            double sum = 0;
            for (double v : prof[mi]) sum += v;
            if (sum > 0)
                for (int x = 0; x < g.E; ++x) H.score[(size_t) x] = prof[mi][(size_t) x] * prior * g.top_k / sum;
        }
        H.total = have_prof ? prof[mi] : std::vector<double>((size_t) g.E, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) g.E * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) g.E * 4, cudaMemcpyHostToDevice));
    }
    const size_t T = (size_t) E.tmax, Kp = (size_t) g.top_k;
    CK(cudaHostAlloc((void**) &E.h_hl, T * g.LAT * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * g.LAT * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_w, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_w2, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, (size_t) g.E * 4, cudaHostAllocDefault));
    E.ecpu = dalloc(T * Kp * g.LAT);
    E.d_miss = dalloc_t<int32_t>(T * Kp);
    E.init_transient();
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming | cudaEventBlockingSync));
    E.pool = std::make_unique<strata::nemotron_h::CpuExperts>(threads);
    CK(cudaDeviceSynchronize());
    std::fprintf(stderr, "strata-nh: hybrid experts: %d of %d per layer in VRAM (%.2f GiB of %.2f GiB), a batch of <= %d rows "
                 "on the GPU + %d CPU threads (PCIe share %.2f, up to %d experts; prefetch %d), bigger chunks streamed through "
                 "%d buffers (%.2f GiB); %.1f s\n", S_, g.E, (double) used / (1 << 30), (double) total / (1 << 30),
                 E.hybrid_t, E.pool->threads(), E.pcie_frac, E.pcie_cap, E.pf_cap, E.xs.nr,
                 (double) E.xs.nr * max_stride / (1 << 30), ms_since(t0) / 1000.0);
    (void) err;
    return true;
}

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == kEos || t == kEos2; }

/// strata-vision's records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats), in prompt order.
bool read_embeddings(const std::string& path, int n_embd, std::vector<float>& rows, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open the embeddings file " + path; return false; }
    rows.clear();
    for (;;) {
        int32_t hd[5];
        f.read((char*) hd, sizeof hd);
        if (f.gcount() == 0) break;
        if (f.gcount() != (std::streamsize) sizeof hd || hd[0] != 0x31455653) { err = "not a strata-vision file"; return false; }
        if (hd[4] != n_embd) {
            err = "the image embeddings are " + std::to_string(hd[4]) + " wide, the model's are " + std::to_string(n_embd) +
                  " (the mmproj of another model?)";
            return false;
        }
        const size_t n = (size_t) hd[1] * (size_t) n_embd;
        const size_t at = rows.size();
        rows.resize(at + n);
        f.read((char*) (rows.data() + at), (std::streamsize) (n * 4));
        if ((size_t) f.gcount() != n * 4) { err = "a truncated embeddings file"; return false; }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ serve
int serve(Engine& E) {
    std::printf("READY %d stop\n", E.ctx);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;             // the tokens whose state the engine holds
    std::vector<std::string> live_keys;    // the pictures among them (GENI img_keys), in order
    std::vector<float> lg, img_rows;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] <file> "
                        "<id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
        Sampling smp;
        std::vector<std::string> keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                std::string s = v;
                size_t a = 0;
                while (a <= s.size()) {
                    const size_t b = s.find(',', a);
                    const std::string k = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
                    if (!k.empty()) keys.push_back(k);
                    if (b == std::string::npos) break;
                    a = b + 1;
                }
            }
        }
        std::string emb_file;
        if (geni) {
            while (*p == ' ') ++p;
            const char* s0 = p;
            while (*p && *p != ' ') ++p;
            emb_file.assign(s0, (size_t) (p - s0));
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.V;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        // the images: their rows at the prompt's <image> tokens, in order
        E.inject.clear();
        size_t n_pad = 0;
        for (int32_t t : ids) n_pad += t == kImagePad;
        if (geni) {
            std::string err;
            if (!read_embeddings(emb_file, E.m->g.N, img_rows, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
            if (img_rows.size() != n_pad * (size_t) E.m->g.N) {
                std::printf("ERR the prompt has %zu image tokens, the embeddings %zu rows\n", n_pad,
                            img_rows.size() / (size_t) E.m->g.N);
                std::fflush(stdout);
                continue;
            }
        }
        stop.store(false);
        // the held state continues when the request extends it (the recurrent state cannot step back); a picture
        // matches by its key
        size_t reuse = 0;
        const bool keys_ok = geni ? keys.size() >= live_keys.size() && std::equal(live_keys.begin(), live_keys.end(), keys.begin())
                                  : live_keys.empty();
        // the request's common prefix with the live sequence: all of it (the request extends it), or a part - then a
        // turn checkpoint at or below that part brings the recurrent state back there
        size_t common = 0;
        while (common < live.size() && common < ids.size() && live[common] == ids[common]) ++common;
        bool restored = false;
        if (keys_ok && live.size() < ids.size() && common == live.size()) reuse = live.size();
        else if (keys_ok && common > 0) {
            const int64_t at = E.ckpt_restore((int64_t) std::min(common, ids.size() - 1));
            if (at > 0) {
                reuse = (size_t) at;
                live.resize(reuse);
                E.ckpt_drop_past(at);
                restored = true;
            }
        }
        if (reuse == 0) {
            E.reset();
            live.clear();
            live_keys.clear();
            E.ckpt_drop_past(0);
        }
        if (restored)
            std::fprintf(stderr, "strata-nh: turn checkpoint: back to %zu tokens of %zu (the rest of the prompt: %zu)\n",
                         reuse, ids.size(), ids.size() - reuse);
        if (geni) {
            size_t r = 0;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == kImagePad) {
                    if (i >= reuse) E.inject[(int64_t) i] = img_rows.data() + r * (size_t) E.m->g.N;
                    ++r;
                }
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        auto pp = [&](int64_t base) {
            return [&, base](int64_t done) {
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) (base + done), (long long) fresh, ms_since(t0),
                            (double) (base + done) / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
            };
        };
        // the prompt but its last token, a turn checkpoint there, then the last token: a request that comes back to
        // this prompt (a regenerated answer) restores the checkpoint and reads one token
        if (fresh > 1) {
            E.process(ids.data() + reuse, fresh - 1, pp(0));
            E.ckpt_save();
            E.process(ids.data() + ids.size() - 1, 1, pp(fresh - 1));
        } else {
            E.process(ids.data() + reuse, fresh, pp(0));
        }
        E.inject.clear();
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        if (geni) live_keys = keys.empty() ? std::vector<std::string>{emb_file} : keys;
        const double prompt_ms = ms_since(t0);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        const bool spec = (E.mtp_d > 0 || E.df) && !smp.needs_logits();
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        bool done = false;
        while (!done) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (spec && E.pos + (E.df ? E.df->block : E.mtp_d + 1) <= E.ctx) {
                const int a = E.df ? E.dflash_step(nx, drafts, picks) : E.spec_step(nx, fix, E.mtp_d, drafts, picks);
                offered += E.df ? (long long) drafts.size() : E.mtp_d;
                accepted += a;
                live.push_back(nx);
                fix.assign(drafts.begin(), drafts.begin() + a);
                for (int i = 0; i < a; ++i) {
                    live.push_back(drafts[(size_t) i]);
                    if (!emit(drafts[(size_t) i])) { done = true; break; }
                }
                nx = picks[a];
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        static int served = 0;
        if (!E.profile_save.empty() && ++served % 8 == 0 && !save_profile(E.profile_save, E.hy, E.m->g.E))
            std::fprintf(stderr, "strata-nh: could not save the expert profile to %s\n", E.profile_save.c_str());
    }
}

}  // namespace

int strata::nemotron_h::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, dump_layers;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 64;
    // the PCIe share and the prefetch are off by default: measured on the RTX 3090 + i7-7800X (DDR4-2133, quad
    // channel) the CPU experts already read RAM at its ceiling (~40 GB/s), and the uploads read the same RAM
    // (docs/NEMOTRON_H.md); a faster PCIe link or a slower CPU is where they pay
    int expert_slots = 0, ring = 64, reserve_mib = 1024, pcie_max = 48, prefetch = 0;
    float pcie_frac = 0.f;
    bool arena = true;
    std::string profile, profile_save, dflash_path, tokens_file;
    int dflash_n = 7;
    int kv_fmt = 0;   // --kv
    int adapt_every = 32, adapt_swaps = 8;
    double expert_vram = -1.0;
    bool warm = true, serve_mode = false, pin = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-nh: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--pool-workers" || a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA_VISIBLE_DEVICES)
        else if (a == "--greedy" || a == "--no-graph" || a == "--no-token-graph") {}
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-nh: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N drafts)\n",
                              v.c_str());
        }
        else if (a == "--dflash") dflash_path = next();
        else if (a == "--dflash-n") dflash_n = std::atoi(next().c_str());
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--stream-ring") ring = std::atoi(next().c_str());
        else if (a == "--pin-experts") pin = true;   // (the arena is pinned already; kept for old command lines)
        else if (a == "--no-arena") arena = false;
        else if (a == "--pcie-frac") pcie_frac = std::strtof(next().c_str(), nullptr);
        else if (a == "--pcie-max") pcie_max = std::atoi(next().c_str());
        else if (a == "--prefetch") prefetch = std::atoi(next().c_str());
        else if (a == "--expert-profile") profile = next();
        else if (a == "--adapt-every") adapt_every = std::max(1, std::atoi(next().c_str()));
        else if (a == "--adapt-swaps") adapt_swaps = std::max(0, std::atoi(next().c_str()));
        else if (a == "--expert-profile-save") profile_save = next();
        else if (a == "--vram-reserve-mib") reserve_mib = std::atoi(next().c_str());
        else if (a == "--nvfp4") {   // native: keep NVFP4 matrices NVFP4; q8 (default): transcode to Q8_0
            const std::string v = next();
            if (v != "native" && v != "q8") { std::fprintf(stderr, "strata-nh: --nvfp4 takes native or q8\n"); return 2; }
            g_nvfp4_native = v == "native";
        }
        else if (a == "--kv") {   // the attention KV cache: fp16 (default), q8_0, q4_0
            const std::string v = next();
            kv_fmt = v == "fp16" || v == "f16" ? 0 : v == "q8_0" || v == "q8" ? 1 : v == "q4_0" || v == "q4" ? 2 : -1;
            if (kv_fmt < 0) { std::fprintf(stderr, "strata-nh: --kv takes fp16, q8_0 or q4_0\n"); return 2; }
        }
        else if (a == "--vision") {}   // the server's: images come as GENI requests, which this engine always takes
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-nh: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-nh --model GGUF [--max-context 32768] [--chunk 1024] [--mtp N] [--expert-vram GiB]\n"
                         "       [--expert-slots N] [--hybrid-rows 64] [--cpu-threads N] [--stream-ring 64] [--pin-experts]\n"
                         "       [--kv fp16|q8_0|q4_0] [--nvfp4 q8|native] [--vram-reserve-mib 1024] [--pcie-frac 0] [--pcie-max 48] [--prefetch 0] [--no-arena]\n"
                         "       [--expert-profile F] [--expert-profile-save F] [--adapt-every 32] [--adapt-swaps 8]\n"
                         "       [--dflash DRAFTER.gguf] [--dflash-n 7] [--tokens-file F (ids; the bench's prompt)]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] [--dump-layers F] |\n"
                         "        --bench-pp N --bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-nh: --model is required\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-nh: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-nh: bad --chunk / --max-context\n"); return 2; }
    if (chunk == 0) chunk = 2048;   // a prompt chunk streams every non-resident expert once: bigger pays it less often
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) { std::fprintf(stderr, "strata-nh: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-nh: %s\n", e.what());
        return 1;
    }
    Dflash drafter;
    if (!dflash_path.empty()) {
        if (mtp_d > 0) { std::fprintf(stderr, "strata-nh: --dflash and --mtp are exclusive\n"); return 2; }
        try {
            if (!load_dflash(dflash_path, m, drafter, err)) { std::fprintf(stderr, "strata-nh: %s\n", err.c_str()); return 1; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-nh: --dflash: %s\n", e.what());
            return 1;
        }
        if (dflash_n < 1 || dflash_n > drafter.block - 1) {
            std::fprintf(stderr, "strata-nh: --dflash-n takes 1 .. %d\n", drafter.block - 1);
            return 2;
        }
    }
    Engine E;
    E.hybrid_t = std::max(1, std::min(hybrid_rows, chunk));
    E.pcie_frac = std::max(0.f, std::min(1.f, pcie_frac));
    E.pcie_cap = std::max(0, pcie_max);
    E.pf_cap = std::max(0, prefetch);
    E.profile_save = profile_save;
    E.adapt_every = adapt_every;
    E.adapt_swaps = adapt_swaps;
    (void) pin;
    E.kv_fmt = kv_fmt;
    E.init(m, ctx, chunk);
    if (!dflash_path.empty()) {
        E.df = &drafter;
        E.df_n = dflash_n;
        E.init_dflash();
    }
    if (!place_experts(m, E, expert_vram, cpu_threads, expert_slots, ring, arena, reserve_mib, profile, err)) {
        std::fprintf(stderr, "strata-nh: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            const long long n = E.hy_cpu + E.hy_gpu + E.tr_pcie + E.tr_pf_hit;
            auto pc = [&](long long v) { return n ? 100.0 * v / n : 0.0; };
            if (E.df && E.df_steps > 0)
                std::fprintf(stderr, "strata-nh: DFlash: %lld steps, %lld of %lld drafts accepted (%.1f%%, %.2f tokens a step); "
                             "draft %.1f ms a step, verify %.1f ms a step; drafter VRAM %.2f GiB\n", E.df_steps, E.df_accepted,
                             E.df_offered, 100.0 * E.df_accepted / std::max(1LL, E.df_offered),
                             1.0 + (double) E.df_accepted / E.df_steps, E.df_draft_ms / E.df_steps, E.df_verify_ms / E.df_steps,
                             (double) E.df_vram / (1 << 30));
            std::fprintf(stderr, "strata-nh: row entries %lld: resident %.1f%%, prefetched %.1f%% (%lld experts sent), PCIe share "
                         "%.1f%%, CPU %.1f%% (%.0f ms; route wait %.0f ms); %lld experts streamed, %lld slot replacements\n", n,
                         pc(E.hy_gpu), pc(E.tr_pf_hit), E.tr_pf_sent, pc(E.tr_pcie), pc(E.hy_cpu), E.hy_cpu_ms,
                         E.t_route_wait_ms, E.hy_streamed, E.hy_swaps);
            if ((E.skip_thr > 0.f || E.skip_rel > 0.f) && E.skip_tot)
                std::fprintf(stderr, "strata-nh: expert skipping (weight < %.3f or < %.3f x row max%s): %lld of %lld CPU-side entries skipped (%.1f%%)\n",
                             E.skip_thr, E.skip_rel, E.skip_renorm ? ", renormalized" : "", E.skip_n, E.skip_tot, 100.0 * E.skip_n / E.skip_tot);
            if (!E.profile_save.empty())
                std::fprintf(stderr, "strata-nh: expert profile %s %s\n", E.profile_save.c_str(),
                             save_profile(E.profile_save, E.hy, E.m->g.E) ? "saved" : "NOT saved");
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (1000 + rng() % 100000u);
        if (!tokens_file.empty()) {   // a real prompt: the file's ids, repeated / cut to bench_pp
            std::ifstream f(tokens_file);
            const std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::string flat = all;
            for (char& c : flat) if (c == '\n' || c == '[' || c == ']' || c == '\r') c = ' ';
            const std::vector<int32_t> ids = parse_ids(flat.c_str());
            if (ids.empty()) { std::fprintf(stderr, "strata-nh: --tokens-file %s has no ids\n", tokens_file.c_str()); return 2; }
            for (size_t i = 0; i < toks.size(); ++i) toks[i] = ids[i % ids.size()];
            std::fprintf(stderr, "strata-nh: bench prompt: %zu ids from %s\n", ids.size(), tokens_file.c_str());
        }
        for (int rep = 0; rep < 2; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            std::vector<int32_t> gen;
            if (E.df) {
                std::vector<int32_t> drafts;
                int32_t picks[16];
                while ((int) gen.size() < bench_tg) {
                    gen.push_back(nx);
                    const int a = E.dflash_step(nx, drafts, picks);
                    for (int i = 0; i < a; ++i) gen.push_back(drafts[(size_t) i]);
                    nx = picks[a];
                }
            } else {
                for (int i = 0; i < bench_tg; ++i) {
                    gen.push_back(nx);
                    const int32_t one = nx;
                    E.process(&one, 1, [](int64_t) {});
                    nx = E.greedy();
                }
            }
            const double tg_ms = ms_since(t1);
            const int n_tg = (int) gen.size();
            std::printf("bench %d gen:", rep);
            for (int t : gen) std::printf(" %d", t);
            std::printf("\n");
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.2f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), n_tg, n_tg > 0 ? (double) n_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-nh: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    if (!dump_layers.empty()) E.dump_f = std::fopen(dump_layers.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    const bool nll_on = std::getenv("STRATA_NH_NLL") != nullptr;   // the forced tokens' mean negative log-likelihood
    double nll_sum = 0;
    long long nll_n = 0, nll_top1 = 0;
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.df && forced.empty() && !ff) {
        std::vector<int32_t> drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.dflash_step(nx, drafts, picks);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-nh: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.2f tok/s), DFlash %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.df_n, E.df_accepted, E.df_offered,
                     E.df_offered ? 100.0 * E.df_accepted / E.df_offered : 0.0);
        return 0;
    }
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-nh: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.2f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered, offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (nll_on && i < (int) forced.size()) {   // teacher-forced log-likelihood of the next real token
            E.host_logits(lg);
            double mx = -1e30;
            for (float v : lg) mx = std::max(mx, (double) v);
            double se = 0;
            for (float v : lg) se += std::exp((double) v - mx);
            nll_sum += (std::log(se) + mx) - (double) lg[(size_t) forced[(size_t) i]];
            ++nll_n;
            if (a == forced[(size_t) i]) ++nll_top1;
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    if (E.dump_f) std::fclose(E.dump_f);
    if (nll_on && nll_n > 0)
        std::fprintf(stderr, "strata-nh: NLL %.5f over %lld forced tokens (perplexity %.4f), top-1 agrees on %.1f%%\n", nll_sum / (double) nll_n, nll_n,
                     std::exp(nll_sum / (double) nll_n), 100.0 * (double) nll_top1 / (double) nll_n);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-nh: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
