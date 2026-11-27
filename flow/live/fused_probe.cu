// Probe: is a hand-written kernel for the last per-point layer worth it?
//
// The last per-point layer of the flow network (256 -> 512, GELU) is 61 % of its arithmetic and produces a
// [128 x 512 points, 512] tensor (64 MB in float16) of which the network only uses the maximum and the mean over the
// points of each hypothesis. This program times that layer in two ways on random data and checks both against a CPU
// reference:
//   unfused  cuBLAS GEMM writes the whole output, a second kernel adds the bias, applies GELU and reduces
//   fused    one kernel (tensor cores through the WMMA interface) computes 64 points x 64 channels at a time, keeps
//            the tile in shared memory, reduces it there, and never writes the output
// The ratio says how much of the layer's time is the memory traffic of the output. Compare "fused" with the time
// TensorRT reports for this layer and its pooling (trtexec --dumpProfile) on the same machine.
//
//   nvcc -O3 -std=c++17 -arch=sm_87 fused_probe.cu -o fused_probe -lcublas      (Jetson Orin; sm_89 for an RTX 4090)
//   ./fused_probe [iterations=50]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int B = 128, P = 512, K = 256, C = 512;   // hypotheses, points, inputs, outputs
constexpr int TP = 64, TC = 64;                     // points and channels of one tile
constexpr int NT = P / TP;                          // point tiles per hypothesis

__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }

// Fused: block = (hypothesis, point tile); 16 warps, warp (wr, wc) computes the 16 x 16 block of points 16 wr.. and
// channels 16 wc.. of the current channel tile. part_max / part_sum: [B, NT, C].
__global__ void fused_layer(const half* __restrict__ A, const half* __restrict__ W, const float* __restrict__ bias,
                            float* __restrict__ part_max, float* __restrict__ part_sum) {
    __shared__ float tile[TP][TC];
    const int b = blockIdx.x / NT, pt = blockIdx.x % NT;
    const int warp = threadIdx.x / 32, wr = warp / 4, wc = warp % 4;
    const half* a0 = A + (size_t(b) * P + size_t(pt) * TP + 16 * wr) * K;
    for (int ct = 0; ct < C / TC; ct++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        const half* w0 = W + size_t(ct * TC + 16 * wc) * K;     // W[channel][input]: column-major (input, channel), leading dimension K
        for (int k = 0; k < K; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, a0 + k, K);
            wmma::load_matrix_sync(fb, w0 + k, K);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(&tile[16 * wr][16 * wc], acc, TC, wmma::mem_row_major);
        __syncthreads();
        if (threadIdx.x < TC) {     // one thread per channel: bias, GELU, maximum and sum over the 64 points
            const int ch = ct * TC + threadIdx.x;
            const float bi = bias[ch];
            float m = -1e30f, s = 0.0f;
            for (int r = 0; r < TP; r++) {
                const float v = gelu(tile[r][threadIdx.x] + bi);
                m = fmaxf(m, v);
                s += v;
            }
            part_max[(size_t(b) * NT + pt) * C + ch] = m;
            part_sum[(size_t(b) * NT + pt) * C + ch] = s;
        }
        __syncthreads();
    }
}

// [B, NT, C] partial results -> maximum and mean [B, C]
__global__ void merge_tiles(const float* __restrict__ part_max, const float* __restrict__ part_sum, float* __restrict__ out_max,
                            float* __restrict__ out_mean) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, ch = i % C;
    float m = -1e30f, s = 0.0f;
    for (int t = 0; t < NT; t++) {
        m = fmaxf(m, part_max[(size_t(b) * NT + t) * C + ch]);
        s += part_sum[(size_t(b) * NT + t) * C + ch];
    }
    out_max[i] = m;
    out_mean[i] = s / float(P);
}

// Unfused second pass: Y [B * P, C] (float16, from the GEMM) -> bias, GELU, maximum and mean over the points
__global__ void reduce_output(const half* __restrict__ Y, const float* __restrict__ bias, float* __restrict__ out_max,
                              float* __restrict__ out_mean) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, ch = i % C;
    const float bi = bias[ch];
    float m = -1e30f, s = 0.0f;
    for (int p = 0; p < P; p++) {
        const float v = gelu(__half2float(Y[(size_t(b) * P + p) * C + ch]) + bi);
        m = fmaxf(m, v);
        s += v;
    }
    out_max[i] = m;
    out_mean[i] = s / float(P);
}

#define CHECK(x) do { if ((x) != 0) { std::fprintf(stderr, "error at %s:%d\n", __FILE__, __LINE__); std::exit(1); } } while (0)

template <class F> double time_ms(int iters, F f) {
    for (int i = 0; i < 5; i++) f();
    CHECK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        CHECK(cudaDeviceSynchronize());
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    std::mt19937 rng(1);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<half> hA(size_t(B) * P * K), hW(size_t(C) * K);
    std::vector<float> fA(hA.size()), fW(hW.size()), fb(C);
    // inputs like the output of a GELU layer (mostly small, non-negative), weights of a trained layer's scale
    for (size_t i = 0; i < hA.size(); i++) { fA[i] = std::max(0.0f, g(rng)) * 0.5f; hA[i] = __float2half(fA[i]); fA[i] = __half2float(hA[i]); }
    for (size_t i = 0; i < hW.size(); i++) { fW[i] = g(rng) / 16.0f; hW[i] = __float2half(fW[i]); fW[i] = __half2float(hW[i]); }
    for (float& v : fb) v = 0.1f * g(rng);

    half *dA, *dW, *dY;
    float *dbias, *dpm, *dps, *dmax, *dmean;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CHECK(cudaMalloc(&dW, hW.size() * sizeof(half)));
    CHECK(cudaMalloc(&dY, size_t(B) * P * C * sizeof(half)));
    CHECK(cudaMalloc(&dbias, C * sizeof(float)));
    CHECK(cudaMalloc(&dpm, size_t(B) * NT * C * sizeof(float)));
    CHECK(cudaMalloc(&dps, size_t(B) * NT * C * sizeof(float)));
    CHECK(cudaMalloc(&dmax, size_t(B) * C * sizeof(float)));
    CHECK(cudaMalloc(&dmean, size_t(B) * C * sizeof(float)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dbias, fb.data(), C * sizeof(float), cudaMemcpyHostToDevice));

    // CPU reference for two hypotheses
    const int check[2] = {0, 77};
    std::vector<float> rmax(2 * C), rmean(2 * C);
    for (int q = 0; q < 2; q++)
        for (int ch = 0; ch < C; ch++) {
            float m = -1e30f;
            double s = 0;
            for (int p = 0; p < P; p++) {
                const float* a = &fA[(size_t(check[q]) * P + p) * K];
                double acc = 0;
                for (int k = 0; k < K; k++) acc += double(a[k]) * fW[size_t(ch) * K + k];
                const float x = float(acc) + fb[ch];
                const float v = 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
                m = std::max(m, v);
                s += v;
            }
            rmax[q * C + ch] = m;
            rmean[q * C + ch] = float(s / P);
        }
    std::vector<float> omax(size_t(B) * C), omean(size_t(B) * C);
    auto compare = [&](const char* name) {
        CHECK(cudaMemcpy(omax.data(), dmax, omax.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(omean.data(), dmean, omean.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float em = 0, ea = 0, scale = 0;
        for (int q = 0; q < 2; q++)
            for (int ch = 0; ch < C; ch++) {
                em = std::max(em, std::fabs(omax[size_t(check[q]) * C + ch] - rmax[q * C + ch]));
                ea = std::max(ea, std::fabs(omean[size_t(check[q]) * C + ch] - rmean[q * C + ch]));
                scale = std::max(scale, std::fabs(rmax[q * C + ch]));
            }
        std::printf("%-8s against the CPU reference: maximum off by at most %.4f, mean by %.5f (values up to %.2f)\n", name, em, ea, scale);
    };

    // unfused: Y^T (C x n, column-major) = W (C x K) A^T (K x n)
    cublasHandle_t h;
    CHECK(cublasCreate(&h));
    const half one = __float2half(1.0f), zero = __float2half(0.0f);
    auto unfused = [&] {
        CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, C, B * P, K, &one, dW, CUDA_R_16F, K, dA, CUDA_R_16F, K, &zero, dY, CUDA_R_16F, C,
                           CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        reduce_output<<<(B * C + 255) / 256, 256>>>(dY, dbias, dmax, dmean);
    };
    auto gemm_only = [&] {
        CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, C, B * P, K, &one, dW, CUDA_R_16F, K, dA, CUDA_R_16F, K, &zero, dY, CUDA_R_16F, C,
                           CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto fused = [&] {
        fused_layer<<<B * NT, 512>>>(dA, dW, dbias, dpm, dps);
        merge_tiles<<<(B * C + 255) / 256, 256>>>(dpm, dps, dmax, dmean);
    };
    unfused();
    CHECK(cudaDeviceSynchronize());
    compare("unfused");
    CHECK(cudaMemset(dmax, 0, size_t(B) * C * sizeof(float)));
    fused();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    compare("fused");

    const double gflop = 2.0 * B * P * double(K) * C / 1e9;
    const double t_g = time_ms(iters, gemm_only), t_u = time_ms(iters, unfused), t_f = time_ms(iters, fused);
    std::printf("layer 256 -> 512 on %d x %d points (%.1f GFLOP), median of %d runs:\n", B, P, gflop, iters);
    std::printf("  cuBLAS GEMM alone (writes %.0f MB)   %7.3f ms   %.2f TFLOP/s\n", double(B) * P * C * 2 / 1e6, t_g, gflop / t_g);
    std::printf("  unfused (GEMM, then reduction)      %7.3f ms   %.2f TFLOP/s\n", t_u, gflop / t_u);
    std::printf("  fused (no output written)           %7.3f ms   %.2f TFLOP/s\n", t_f, gflop / t_f);
    return 0;
}
