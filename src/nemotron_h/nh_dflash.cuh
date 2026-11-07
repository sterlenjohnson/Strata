// src/nemotron_h/nh_dflash.cuh - the kernels of the DFlash + LiLiCorr drafter (Nemotron 3.5 Bolt's
// dflash-*.gguf, GGUF architecture `dflash` with `lilicorr.*`).  The formulas are upstream llama.cpp's
// (src/models/dflash.cpp: build_dflash2_conv, the non-causal SWA decoder with attention sinks, build_lilicorr_scores;
// common/speculative.cpp: the greedy walk over the candidate lattice).  Included by nh_engine.cu only.
#pragma once

#include "nh_kernels.cuh"

namespace nh {

constexpr int kDfMaxKeys = 1024 + 16;   // a draft query's keys: the sliding window's context + the block

__global__ void k_df_add(float* x, const float* a, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

/// x[t][i] = act(x[t][i] + b[i]); act 0: none, 1: SiLU
__global__ void k_df_bias_act(float* x, const float* b, int n, int64_t total, int act) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i] + (b ? b[i % n] : 0.f);
    if (act == 1) v = v / (1.f + __expf(-v));
    x[i] = v;
}

/// g = silu(g) * u
__global__ void k_df_swiglu(float* g, const float* u, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    g[i] = v / (1.f + __expf(-v)) * u[i];
}

/// The grouped dynamic convolution over a block's rows (build_dflash2_conv, one side):
/// out[t][c] = sum_{tap < K, t - tap >= 0} (base[tap][c] + dyn[t][side][tap][c / group]) * in[t - tap][c]
/// base: the side's [K][H] plane; dyn rows [2][K][H / group] (ld dyn_ld).
__global__ void k_df_conv(const float* in, const float* dyn, int dyn_ld, const float* base, int side, float* out, int rows,
                          int H, int group, int K) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * H) return;
    const int t = (int) (i / H), c = (int) (i % H);
    const int groups = H / group;
    float acc = 0.f;
    for (int tap = 0; tap < K; ++tap) {
        if (t - tap < 0) break;
        const float w = base[(size_t) tap * H + c] + dyn[(size_t) t * dyn_ld + (size_t) side * K * groups + tap * groups + c / group];
        acc = fmaf(w, in[(size_t) (t - tap) * H + c], acc);
    }
    out[i] = acc;
}

/// Per (row, head): RMS norm over the 128 dims times w, then NeoX rope at pos0 + row (theta = base^(-2i/128)).
/// grid (rows, heads), 128 threads.  In place on x [rows][ld] (head h at h * 128).
__global__ void k_df_norm_rope(float* x, int ld, const float* w, float eps, int64_t pos0, float base) {
    __shared__ float sh[128];
    __shared__ float v[128];
    float* r = x + (size_t) blockIdx.x * ld + (size_t) blockIdx.y * 128;
    const int i = threadIdx.x;
    const float a = r[i];
    sh[i] = a * a;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) sh[i] += sh[i + s];
        __syncthreads();
    }
    const float rr = rsqrtf(sh[0] / 128.f + eps);
    v[i] = a * rr * w[i];
    __syncthreads();
    const int j = i & 63;
    const double p = (double) (pos0 + blockIdx.x);
    const double ang = p * pow((double) base, -(double) (2 * j) / 128.0);
    const float c = (float) cos(ang), s = (float) sin(ang);
    // NeoX: pairs (j, j + 64)
    r[i] = i < 64 ? v[j] * c - v[j + 64] * s : v[j] * s + v[j + 64] * c;
}

/// The context update: K (already normed + roped) and V rows [rows][HKV * 128] into the ring cells (pos0 + row) % R,
/// FP16, [cell][HKV][128].  grid (rows, HKV), 128 threads.
__global__ void k_df_ring_store(const float* k, const float* v, __half* kr, __half* vr, int HKV, int64_t pos0, int R) {
    const int row = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    const size_t cell = (size_t) ((pos0 + row) % R);
    const size_t src = (size_t) row * HKV * 128 + (size_t) hh * 128 + i;
    const size_t dst = (cell * HKV + hh) * 128 + i;
    kr[dst] = __float2half(k[src]);
    vr[dst] = __float2half(v[src]);
}

/// The draft block's attention (non-causal inside the block, sliding window over the context, attention sinks):
/// query row r (position s + r) of head h sees the context cells at positions [max(0, s + r - (W - 1)), s) from the
/// ring and all `nb` block rows.  grid (nb, H), 128 threads.  q [nb][H*128], kb/vb [nb][HKV*128], out [nb][H*128].
__global__ void __launch_bounds__(128) k_df_attn(const float* q, const float* kb, const float* vb, const __half* kr,
                                                 const __half* vr, const float* sinks, int H, int HKV, int nb, int64_t s,
                                                 int R, int W, float scale, float* out) {
    __shared__ float qs[128];
    __shared__ float sc[kDfMaxKeys];
    __shared__ float red[128];
    const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int kvh = h / (H / HKV);
    qs[tid] = q[(size_t) r * H * 128 + (size_t) h * 128 + tid];
    const int64_t lo = s + r - (W - 1) > 0 ? s + r - (W - 1) : 0;
    const int nctx = (int) (s - lo);
    const int nk = nctx + nb;
    __syncthreads();
    for (int k = tid; k < nk; k += 128) {
        float d = 0.f;
        if (k < nctx) {
            const size_t cell = (size_t) ((lo + k) % R);
            const __half* kk = kr + (cell * HKV + kvh) * 128;
            for (int i = 0; i < 128; ++i) d = fmaf(qs[i], __half2float(kk[i]), d);
        } else {
            const float* kk = kb + (size_t) (k - nctx) * HKV * 128 + (size_t) kvh * 128;
            for (int i = 0; i < 128; ++i) d = fmaf(qs[i], kk[i], d);
        }
        sc[k] = d * scale;
    }
    __syncthreads();
    float mx = sinks ? sinks[h] : -INFINITY;
    for (int k = tid; k < nk; k += 128) mx = fmaxf(mx, sc[k]);
    red[tid] = mx;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int k = tid; k < nk; k += 128) {
        const float e = __expf(sc[k] - mx);
        sc[k] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float tot = red[0] + (sinks ? __expf(sinks[h] - mx) : 0.f);
    float acc = 0.f;
    for (int k = 0; k < nk; ++k) {
        float vv;
        if (k < nctx) vv = __half2float(vr[(((size_t) ((lo + k) % R)) * HKV + kvh) * 128 + tid]);
        else vv = vb[(size_t) (k - nctx) * HKV * 128 + (size_t) kvh * 128 + tid];
        acc = fmaf(sc[k], vv, acc);
    }
    out[(size_t) r * H * 128 + (size_t) h * 128 + tid] = acc / tot;
}

// ---------------------------------------------------------------------------------------------- LiLiCorr
/// The candidate features of each node: LayerNorm (eps 1e-5) of its 5 features with weight/bias, then
/// SiLU(mlp_1 x + b1) [5 -> hidden].  grid nodes, 256 threads.
__global__ void k_lc_feat(const float* feat, const float* nw, const float* nb, const float* w1, const float* b1, float* out,
                          int hidden) {
    const int n = blockIdx.x;
    float f[5];
    float mean = 0.f;
    for (int i = 0; i < 5; ++i) { f[i] = feat[n * 5 + i]; mean += f[i]; }
    mean /= 5.f;
    float var = 0.f;
    for (int i = 0; i < 5; ++i) var += (f[i] - mean) * (f[i] - mean);
    const float rs = rsqrtf(var / 5.f + 1e-5f);
    for (int i = 0; i < 5; ++i) f[i] = (f[i] - mean) * rs * nw[i] + nb[i];
    for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float v = b1[o];
        for (int i = 0; i < 5; ++i) v = fmaf(w1[(size_t) o * 5 + i], f[i], v);
        out[(size_t) n * hidden + o] = v / (1.f + __expf(-v));
    }
}

/// x[node] = tok[node] + pass[slot] + feat[node] + slot_emb[slot] + rank_emb[rank] (node = slot * topk + rank)
__global__ void k_lc_combine(const float* tok, const float* pass, const float* feat, const float* slot_e, const float* rank_e,
                             float* x, int hidden, int topk, int nodes) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) nodes * hidden) return;
    const int n = (int) (i / hidden), c = (int) (i % hidden), s = n / topk, r = n % topk;
    x[i] = tok[i] + pass[(size_t) s * hidden + c] + feat[i] + slot_e[(size_t) s * hidden + c] + rank_e[(size_t) r * hidden + c];
}

/// The head's attention over all nodes (no mask but the learned bias[h][q][k]); qkv rows [nodes][3 * hidden]
/// (q | k | v, head h at h * hd).  grid (nodes, heads), hd threads (128).
__global__ void k_lc_attn(const float* qkv, const float* bias, int nodes, int hidden, int hd, float scale, float* out) {
    __shared__ float qs[128];
    __shared__ float sc[64];
    const int qn = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, H = gridDim.y;
    const size_t ld = (size_t) 3 * hidden;
    qs[tid] = qkv[(size_t) qn * ld + (size_t) h * hd + tid];
    __syncthreads();
    if (tid < nodes) {
        const float* kk = qkv + (size_t) tid * ld + hidden + (size_t) h * hd;
        float d = 0.f;
        for (int i = 0; i < hd; ++i) d = fmaf(qs[i], kk[i], d);
        sc[tid] = d * scale + bias[((size_t) h * nodes + qn) * nodes + tid];
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < nodes; ++k) mx = fmaxf(mx, sc[k]);
        float s = 0.f;
        for (int k = 0; k < nodes; ++k) { sc[k] = __expf(sc[k] - mx); s += sc[k]; }
        for (int k = 0; k < nodes; ++k) sc[k] /= s;
    }
    __syncthreads();
    float acc = 0.f;
    for (int k = 0; k < nodes; ++k) acc = fmaf(sc[k], qkv[(size_t) k * ld + 2 * hidden + (size_t) h * hd + tid], acc);
    out[(size_t) qn * hidden + (size_t) h * hd + tid] = acc;
    (void) H;
}

/// factor_in[node] = [x | a | x * a] (a: the one anchor row)
__global__ void k_lc_factor_in(const float* x, const float* a, float* out, int hidden, int nodes) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) nodes * hidden) return;
    const int n = (int) (i / hidden), c = (int) (i % hidden);
    const float xv = x[i], av = a[c];
    float* o = out + (size_t) n * 3 * hidden;
    o[c] = xv;
    o[hidden + c] = av;
    o[2 * hidden + c] = xv * av;
}

/// rows x [rows][n] /= max(||x||, eps) (ggml_l2_norm); one block of 256 per row
__global__ void k_lc_l2(float* x, int n, float eps) {
    __shared__ float sh[kThreads];
    float* r = x + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    s = block_sum<kThreads>(s, sh);
    const float sc = 1.f / fmaxf(sqrtf(s), eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] *= sc;
}

/// The lattice scores: out[0 .. topk) = ls * in[slot 0][j] . anchor_out; then for slots s = 1 .. slots-1 the topk x topk
/// block [p][j] = ls * in[s][j] . out_[s-1][p].  One block of 256 per (slot, p) pair (slot 0: p = 0 only), warp per j.
__global__ void k_lc_scores(const float* in, const float* outv, const float* anc, int topk, int fd, float ls, float* scores) {
    const int s = blockIdx.x, p = blockIdx.y;
    if (s == 0 && p > 0) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int j = warp; j < topk; j += blockDim.x / 32) {
        const float* a = in + ((size_t) s * topk + j) * fd;
        const float* b = s == 0 ? anc : outv + ((size_t) (s - 1) * topk + p) * fd;
        float d = 0.f;
        for (int i = lane; i < fd; i += 32) d = fmaf(a[i], b[i], d);
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) {
            if (s == 0) scores[j] = ls * d;
            else scores[topk + ((size_t) (s - 1) * topk + p) * topk + j] = ls * d;
        }
    }
}

}  // namespace nh
