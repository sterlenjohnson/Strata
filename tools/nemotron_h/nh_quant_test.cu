// tools/nemotron_h/nh_quant_test.cu - unit checks of the nh path's quantized KV cache (--kv q8_0 / q4_0) against
// ggml-cpu's reference rounding (quantize_row_q8_0_ref / quantize_row_q4_0_ref, transcribed) and a double-precision
// attention over the reference-dequantized cache.  Built by hand (nvcc -arch=sm_89 -O2 -I src/nemotron_h).
#include "nh_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace nh;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

static float h2f(__half h) { return __half2float(h); }

// ggml quantize_row_q8_0_ref / q4_0_ref on one 32-block; codes in this cache's order
static void ref_q8(const float* x, int8_t* q, __half* d_out) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.f, id = d ? 1.0f / d : 0.0f;
    *d_out = __float2half(d);
    for (int i = 0; i < 32; ++i) q[i] = (int8_t) roundf(x[i] * id);
}
static void ref_q4(const float* x, uint8_t* q, __half* d_out) {
    float amax = 0.f, mx = 0.f;
    for (int i = 0; i < 32; ++i)
        if (amax < std::fabs(x[i])) { amax = std::fabs(x[i]); mx = x[i]; }
    const float d = mx / -8.f, id = d ? 1.0f / d : 0.0f;
    *d_out = __float2half(d);
    for (int i = 0; i < 16; ++i) {
        const uint8_t a = (uint8_t) std::min(15, (int) (int8_t) (x[2 * i] * id + 8.5f));
        const uint8_t b = (uint8_t) std::min(15, (int) (int8_t) (x[2 * i + 1] * id + 8.5f));
        q[i] = a | (b << 4);
    }
}

int run(int q4) {
    const int H = 32, HKV = 2, ctx = 4096, T = 3000;   // T cells stored at once (pos0 = 0)
    const size_t cells = (size_t) HKV * ctx, cb = q4 ? kHD / 2 : kHD;
    std::mt19937 rng(1234 + q4);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> kx((size_t) T * HKV * kHD), vx(kx.size()), q((size_t) H * kHD);
    for (size_t i = 0; i < kx.size(); ++i) {
        kx[i] = nd(rng) * (i % 97 == 0 ? 20.f : 2.f);   // a few outliers per block, like real K
        vx[i] = nd(rng) * 0.5f;
    }
    for (auto& v : q) v = nd(rng);
    float *dk, *dv, *dq, *pacc, *pm, *pl, *out;
    uint8_t *kq, *vq;
    __half *ks, *vs, *deq;
    int32_t* pos0;
    const int n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
    CK(cudaMalloc(&dk, kx.size() * 4)); CK(cudaMalloc(&dv, vx.size() * 4)); CK(cudaMalloc(&dq, q.size() * 4));
    CK(cudaMalloc(&kq, cells * cb)); CK(cudaMalloc(&vq, cells * cb));
    CK(cudaMalloc(&ks, cells * 8)); CK(cudaMalloc(&vs, cells * 8));
    CK(cudaMalloc(&deq, (size_t) ctx * kHD * 2));
    CK(cudaMalloc(&pacc, (size_t) H * n_chunk * kHD * 4)); CK(cudaMalloc(&pm, H * n_chunk * 4)); CK(cudaMalloc(&pl, H * n_chunk * 4));
    CK(cudaMalloc(&out, (size_t) H * kHD * 4)); CK(cudaMalloc(&pos0, 4));
    CK(cudaMemcpy(dk, kx.data(), kx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vx.data(), vx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(pos0, 0, 4));
    if (q4) k_kv_store_q<1><<<dim3(HKV, T), kHD>>>(dk, dv, kq, ks, vq, vs, HKV, pos0, ctx);
    else k_kv_store_q<0><<<dim3(HKV, T), kHD>>>(dk, dv, kq, ks, vq, vs, HKV, pos0, ctx);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hkq(cells * cb), hvq(cells * cb);
    std::vector<__half> hks(cells * 4), hvs(cells * 4);
    CK(cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hks.data(), ks, hks.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hvs.data(), vs, hvs.size() * 2, cudaMemcpyDeviceToHost));
    // 1. codes and scales bitwise against the reference; the reference dequant kept for 2 / 3
    long long bad = 0;
    std::vector<float> Kd(cells * kHD, 0.f), Vd(cells * kHD, 0.f);
    double kerr = 0, kref = 0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < HKV; ++h) {
            const size_t cell = (size_t) h * ctx + t;
            for (int w = 0; w < 2; ++w) {
                const float* x = (w ? vx.data() : kx.data()) + ((size_t) t * HKV + h) * kHD;
                const uint8_t* gq = (w ? hvq.data() : hkq.data()) + cell * cb;
                const __half* gs = (w ? hvs.data() : hks.data()) + cell * 4;
                float* D = (w ? Vd.data() : Kd.data()) + cell * kHD;
                for (int b = 0; b < 4; ++b) {
                    uint8_t rq[32];
                    __half rd;
                    if (q4) ref_q4(x + b * 32, rq, &rd); else ref_q8(x + b * 32, (int8_t*) rq, &rd);
                    if (std::memcmp(&rd, gs + b, 2) || std::memcmp(rq, gq + b * (cb / 4), cb / 4)) ++bad;
                    for (int i = 0; i < 32; ++i) {
                        const float v = q4 ? (float) (((rq[i / 2] >> (4 * (i & 1))) & 15) - 8) * h2f(rd)
                                           : (float) (int8_t) rq[i] * h2f(rd);
                        D[b * 32 + i] = v;
                        if (!w) { kerr += (v - x[b * 32 + i]) * (double) (v - x[b * 32 + i]); kref += (double) x[b * 32 + i] * x[b * 32 + i]; }
                    }
                }
            }
        }
    std::printf("[%s] store: %lld of %d blocks differ from ggml's reference (codes or scale); K rel-RMS quant error %.4f\n",
                q4 ? "q4_0" : "q8_0", bad, T * HKV * 2 * 4, std::sqrt(kerr / kref));
    // 2. the prompt's dequant kernel against the reference dequant (FP16 of it: bitwise)
    long long dbad = 0;
    std::vector<__half> hd((size_t) ctx * kHD);
    for (int h = 0; h < HKV; ++h) {
        if (q4) k_kv_deq<1><<<T, kHD>>>(vq, vs, (size_t) h * ctx, deq); else k_kv_deq<0><<<T, kHD>>>(vq, vs, (size_t) h * ctx, deq);
        CK(cudaMemcpy(hd.data(), deq, (size_t) T * kHD * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t) T * kHD; ++i) {
            const __half r = __float2half(Vd[(size_t) h * ctx * kHD + i]);
            if (std::memcmp(&r, &hd[i], 2)) ++dbad;
        }
    }
    std::printf("[%s] dequant: %lld of %d values differ\n", q4 ? "q4_0" : "q8_0", dbad, T * HKV * kHD);
    // 3. decode attention (one query row at position T-1) against double attention over the reference dequant
    int p = T - 1;
    CK(cudaMemcpy(pos0, &p, 4, cudaMemcpyHostToDevice));
    const float scale = 1.0f / std::sqrt((float) kHD);
    const int nc = (T + kChunkCells - 1) / kChunkCells;
    if (q4) k_attn_part_q<1><<<dim3(nc, H), kThreads>>>(dq, kq, ks, vq, vs, pos0, 0, H, HKV, ctx, scale, n_chunk, pacc, pm, pl);
    else k_attn_part_q<0><<<dim3(nc, H), kThreads>>>(dq, kq, ks, vq, vs, pos0, 0, H, HKV, ctx, scale, n_chunk, pacc, pm, pl);
    k_attn_merge<<<H, kHD>>>(pacc, pm, pl, pos0, 0, H, n_chunk, out);
    std::vector<float> go((size_t) H * kHD);
    CK(cudaMemcpy(go.data(), out, go.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0, maxrel_x = 0;
    for (int h = 0; h < H; ++h) {
        const int kvh = h / (H / HKV);
        for (int pass = 0; pass < 2; ++pass) {   // 0: over the dequantized cache; 1: over the exact fp32 K / V
            std::vector<double> s(T);
            double mx = -1e300;
            for (int c = 0; c < T; ++c) {
                double a = 0;
                for (int i = 0; i < kHD; ++i) {
                    const double kv = pass ? kx[((size_t) c * HKV + kvh) * kHD + i] : Kd[((size_t) kvh * ctx + c) * kHD + i];
                    a += (double) q[(size_t) h * kHD + i] * kv;
                }
                s[c] = a * scale;
                mx = std::max(mx, s[c]);
            }
            double den = 0;
            for (int c = 0; c < T; ++c) den += (s[c] = std::exp(s[c] - mx));
            double num2 = 0, err2 = 0;
            for (int i = 0; i < kHD; ++i) {
                double o = 0;
                for (int c = 0; c < T; ++c)
                    o += s[c] * (pass ? vx[((size_t) c * HKV + kvh) * kHD + i] : Vd[((size_t) kvh * ctx + c) * kHD + i]);
                o /= den;
                num2 += o * o;
                err2 += (o - go[(size_t) h * kHD + i]) * (o - go[(size_t) h * kHD + i]);
            }
            (pass ? maxrel_x : maxrel) = std::max(pass ? maxrel_x : maxrel, std::sqrt(err2 / num2));
        }
    }
    std::printf("[%s] decode attention: max rel-RMS vs double over the dequantized cache %.2e (kernel arithmetic); vs "
                "exact fp32 K/V %.2e (quantization)\n", q4 ? "q4_0" : "q8_0", maxrel, maxrel_x);
    const bool ok = bad == 0 && dbad == 0 && maxrel < 1e-5;
    std::printf("[%s] %s\n", q4 ? "q4_0" : "q8_0", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() { return run(0) | run(1); }
