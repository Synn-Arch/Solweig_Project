// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_residency.cu — T13 residency/graphs runtime ABI (TASKS T13, DESIGN
// 13.3/13.4). Additive to the T12 ABI; sw_api.cu is untouched except for
// the monotonic allocation counters in sw_alloc.
//
// Surfaces:
//   * pinned host memory (cudaMallocHost) — the staging area for async
//     D2H output leases;
//   * one stream + cudaEvents — every copy/kernel/download is enqueued
//     on the caller's stream in submission order; events fence the
//     publish-before-completion and buffer-reuse hazards;
//   * 2D async copies — row-chunk plane upload/download without pitch
//     games (width == pitch == plane row bytes);
//   * stream-parameterized launches of the EXISTING T12 kernels
//     (sw_rad_day_kernel / sw_rad_night_kernel) plus the graph-bucket
//     UTCI variant whose logical n is read from device memory, so an
//     instantiated CUDA graph is replayable across a whole capacity
//     bucket;
//   * graph capture/instantiate/launch/destroy — capture is stream
//     capture (cudaStreamBeginCapture, ThreadLocal mode); the pinned
//     n-word memcpy enqueued before the kernel becomes a memcpy NODE,
//     refreshed from pinned memory on every replay.
//
// Every entry point returns 0 on success, non-zero CUDA error otherwise;
// sw_last_error-style messages go to this TU's local buffer, retrievable
// via sw_res_last_error().
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#include "../include/sw_strict_math.cuh"
#include "../include/sw_kernels_decl.cuh"

static char g_res_last_error[1024] = "no error";
static std::mutex g_res_mutex;

#define SW_RES_CHECK(expr, what)                                                \
    do {                                                                        \
        cudaError_t err_ = (expr);                                              \
        if (err_ != cudaSuccess) {                                              \
            snprintf(g_res_last_error, sizeof g_res_last_error, "%s: %s", what, \
                     cudaGetErrorString(err_));                                 \
            return (int)err_;                                                   \
        }                                                                       \
    } while (0)

extern "C" {

const char* sw_res_last_error() { return g_res_last_error; }

// --- pinned host memory -----------------------------------------------------
int sw_pinned_alloc(void** out, long long bytes) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaMallocHost(out, (size_t)bytes), "cudaMallocHost");
    return 0;
}

int sw_pinned_free(void* p) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaFreeHost(p), "cudaFreeHost");
    return 0;
}

// --- streams & events ---------------------------------------------------------
int sw_stream_create(void** out) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaStreamCreate((cudaStream_t*)out), "cudaStreamCreate");
    return 0;
}

int sw_stream_destroy(void* s) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaStreamDestroy((cudaStream_t)s), "cudaStreamDestroy");
    return 0;
}

int sw_stream_sync(void* s) {
    SW_RES_CHECK(cudaStreamSynchronize((cudaStream_t)s),
                 "cudaStreamSynchronize");
    return 0;
}

int sw_event_create(void** out) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaEventCreate((cudaEvent_t*)out), "cudaEventCreate");
    return 0;
}

int sw_event_destroy(void* e) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaEventDestroy((cudaEvent_t)e), "cudaEventDestroy");
    return 0;
}

int sw_event_record(void* e, void* s) {
    SW_RES_CHECK(cudaEventRecord((cudaEvent_t)e, (cudaStream_t)s),
                 "cudaEventRecord");
    return 0;
}

// 0 = complete, 1 = not ready (still pending), negative = error
int sw_event_query(void* e) {
    cudaError_t err = cudaEventQuery((cudaEvent_t)e);
    if (err == cudaSuccess) return 0;
    if (err == cudaErrorNotReady) return 1;
    snprintf(g_res_last_error, sizeof g_res_last_error, "cudaEventQuery: %s",
             cudaGetErrorString(err));
    return -(int)err;
}

int sw_event_sync(void* e) {
    SW_RES_CHECK(cudaEventSynchronize((cudaEvent_t)e), "cudaEventSynchronize");
    return 0;
}

// --- async copies (plane row-chunks) -----------------------------------------
// width == pitch on both sides: planes are contiguous, a chunk at row r0
// is just base + r0*width. Pageable host sources follow CUDA semantics
// (the call returns once the source has been staged/consumed), so the
// caller's numpy chunk is safe to reuse after the call returns.
int sw_copy2d_h2d_async(const void* host, void* dev, long long width,
                        long long n_rows, void* stream) {
    SW_RES_CHECK(cudaMemcpy2DAsync(dev, (size_t)width, host, (size_t)width,
                                   (size_t)width, (size_t)n_rows,
                                   cudaMemcpyHostToDevice,
                                   (cudaStream_t)stream),
                 "Memcpy2DAsyncH2D");
    return 0;
}

int sw_copy2d_d2h_async(const void* dev, void* host, long long width,
                        long long n_rows, void* stream) {
    SW_RES_CHECK(cudaMemcpy2DAsync(host, (size_t)width, dev, (size_t)width,
                                   (size_t)width, (size_t)n_rows,
                                   cudaMemcpyDeviceToHost,
                                   (cudaStream_t)stream),
                 "Memcpy2DAsyncD2H");
    return 0;
}

int sw_copy_h2d_async(const void* host, void* dev, long long bytes,
                      void* stream) {
    SW_RES_CHECK(cudaMemcpyAsync(dev, host, (size_t)bytes,
                                 cudaMemcpyHostToDevice,
                                 (cudaStream_t)stream),
                 "MemcpyAsyncH2D");
    return 0;
}

int sw_copy_d2h_async(const void* dev, void* host, long long bytes,
                      void* stream) {
    SW_RES_CHECK(cudaMemcpyAsync(host, dev, (size_t)bytes,
                                 cudaMemcpyDeviceToHost,
                                 (cudaStream_t)stream),
                 "MemcpyAsyncD2H");
    return 0;
}

// --- device memory info -------------------------------------------------------
int sw_mem_info(long long* free_bytes, long long* total_bytes) {
    size_t free_ = 0, total_ = 0;
    SW_RES_CHECK(cudaMemGetInfo(&free_, &total_), "cudaMemGetInfo");
    *free_bytes = (long long)free_;
    *total_bytes = (long long)total_;
    return 0;
}

// --- stream-parameterized kernel launches ------------------------------------
// Identical argument order to sw_api.cu's sw_run_rad_day/sw_run_rad_night;
// the trailing stream parameter is the ONLY difference, plus the explicit
// grid (the T12 macro computes the same (n+255)/256 geometry).
int sw_res_rad_day(
    const float* buildings, const float* aspect, const float* wallbol,
    const float* alb_grid, const float* emis_grid, const float* svfbuveg,
    const float* diffsh, const unsigned char* sh_pb,
    const unsigned char* veg_pb, const unsigned char* vbsh_pb,
    const unsigned char* sun_pb, const unsigned char* shd_pb,
    const long long* dp_rank, const signed char* guard_true,
    const float* shadow, const float* sunwall, const float* albshadow,
    const float* alb, const float* Lup_pre, const float* gvflup_extra,
    const float* lv2, const float* ster, const float* psin, const float* pcos,
    const float* lumChi, const float* lsky_d2, const float* lsky_s2,
    const signed char* card_e, const signed char* card_s,
    const signed char* card_w, const signed char* card_n,
    const float* ccos_e, const float* ccos_s, const float* ccos_w,
    const float* ccos_n, const float* walk_az_low, const float* walk_az_high,
    const int* walk_az_branch, const int* walk_dy, const int* walk_dx,
    const signed char* jE, const signed char* jS, const signed char* jW,
    const signed char* jN, const float* F_sh, const float* Tg_plane,
    const float* m_lup_in, const float* m_e_in, const float* m_s_in,
    const float* m_w_in, const float* m_n_in, const float* m_tg_in,
    float* o_tmrt, float* o_kdown, float* o_kup, float* o_ldown, float* o_lup,
    float* o_ke, float* o_ks, float* o_kw, float* o_kn, float* o_le,
    float* o_ls, float* o_lw, float* o_ln, float* o_ksidei, float* o_tgout,
    float* o_lside, float* o_ksided, float* o_drad, float* o_kside,
    float* n_lup, float* n_e, float* n_s, float* n_w, float* n_n, float* n_tg,
    float ks_sun, float ks_shd, float radI, float radD, float radG,
    float sinalt, float cosalt, double veg64, double shd64, double sun64,
    float ta273, float Lwall32, float Ta32,
    float w1_0, float w1_1, float w1_2, float w1_3, float w1_4, float w1_5,
    int rows, int cols, int n_patches, int kside_n, int fd_eq_1, int branch2,
    int sun_stride, void* stream) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    long long n = (long long)rows * cols;
    long long blocks = (n + 255) / 256;
    if (blocks < 1) blocks = 1;
    sw_rad_day_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        buildings, aspect, wallbol, alb_grid, emis_grid, svfbuveg, diffsh,
        sh_pb, veg_pb, vbsh_pb, sun_pb, shd_pb, dp_rank, guard_true, shadow,
        sunwall, albshadow, alb, Lup_pre, gvflup_extra, lv2, ster, psin, pcos,
        lumChi, lsky_d2, lsky_s2, card_e, card_s, card_w, card_n, ccos_e,
        ccos_s, ccos_w, ccos_n, walk_az_low, walk_az_high, walk_az_branch,
        walk_dy, walk_dx, jE, jS, jW, jN, F_sh, Tg_plane, m_lup_in, m_e_in,
        m_s_in, m_w_in, m_n_in, m_tg_in, o_tmrt, o_kdown, o_kup, o_ldown,
        o_lup, o_ke, o_ks, o_kw, o_kn, o_le, o_ls, o_lw, o_ln, o_ksidei,
        o_tgout, o_lside, o_ksided, o_drad, o_kside, n_lup, n_e, n_s, n_w,
        n_n, n_tg, ks_sun, ks_shd, radI, radD, radG, sinalt, cosalt, veg64,
        shd64, sun64, ta273, Lwall32, Ta32, w1_0, w1_1, w1_2, w1_3, w1_4,
        w1_5, rows, cols, n_patches, kside_n, fd_eq_1, branch2, sun_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_res_last_error, sizeof g_res_last_error,
                 "launch sw_rad_day_kernel: %s", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int sw_res_rad_night(
    const unsigned char* sh_pb, const unsigned char* veg_pb,
    const unsigned char* vbsh_pb, const float* night_Lup,
    const float* ster, const float* psin, const float* pcos,
    const float* lsky_d2, const float* lsky_s2,
    const signed char* card_e, const signed char* card_s,
    const signed char* card_w, const signed char* card_n,
    const float* ccos_e, const float* ccos_s, const float* ccos_w,
    const float* ccos_n, double veg64, double shd64,
    float* o_tmrt, float* o_ldown, float* o_lside, float* o_le, float* o_ls,
    float* o_lw, float* o_ln,
    int rows, int cols, int n_patches, void* stream) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    long long n = (long long)rows * cols;
    long long blocks = (n + 255) / 256;
    if (blocks < 1) blocks = 1;
    sw_rad_night_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        sh_pb, veg_pb, vbsh_pb, night_Lup, ster, psin, pcos, lsky_d2, lsky_s2,
        card_e, card_s, card_w, card_n, ccos_e, ccos_s, ccos_w, ccos_n, veg64,
        shd64, o_tmrt, o_ldown, o_lside, o_le, o_ls, o_lw, o_ln, rows, cols,
        n_patches);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_res_last_error, sizeof g_res_last_error,
                 "launch sw_rad_night_kernel: %s", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

// Graph-bucket UTCI: grid is FIXED by the capacity bucket; the logical
// extent n is read from n_dev[0] (refreshed by the graph's memcpy node
// per replay). Lanes k >= n are bucket padding: the guard returns before
// ANY input read or output write (T13 RED witness 2).
int sw_res_utci_sparse_bucket(const float* ta, const float* rh,
                              const float* tmrt, const float* va, float* out,
                              const long long* n_dev, int T, int grain,
                              unsigned grid_blocks, void* stream) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    sw_utci_sparse_bucket_kernel<<<grid_blocks, 256, 0,
                                   (cudaStream_t)stream>>>(
        ta, rh, tmrt, va, out, n_dev, T, grain);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_res_last_error, sizeof g_res_last_error,
                 "launch sw_utci_sparse_bucket_kernel: %s",
                 cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

// --- CUDA graphs ---------------------------------------------------------------
// Stream capture: the caller begins capture on its stream, enqueues the
// pinned n-word memcpy + kernel launch through the ABI above, ends
// capture and instantiates. Replay via cudaGraphLaunch re-executes the
// recorded nodes with CURRENT pinned-memory contents.
int sw_graph_begin_capture(void* stream) {
    SW_RES_CHECK(cudaStreamBeginCapture((cudaStream_t)stream,
                                        cudaStreamCaptureModeThreadLocal),
                 "cudaStreamBeginCapture");
    return 0;
}

int sw_graph_end_capture(void* stream, void** graph_out) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaStreamEndCapture((cudaStream_t)stream,
                                      (cudaGraph_t*)graph_out),
                 "cudaStreamEndCapture");
    return 0;
}

int sw_graph_instantiate(void* graph, void** exec_out) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaGraphInstantiate((cudaGraphExec_t*)exec_out,
                                      (cudaGraph_t)graph, 0),
                 "cudaGraphInstantiate");
    return 0;
}

int sw_graph_launch(void* exec, void* stream) {
    SW_RES_CHECK(cudaGraphLaunch((cudaGraphExec_t)exec,
                                 (cudaStream_t)stream),
                 "cudaGraphLaunch");
    return 0;
}

int sw_graph_exec_destroy(void* exec) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaGraphExecDestroy((cudaGraphExec_t)exec),
                 "cudaGraphExecDestroy");
    return 0;
}

int sw_graph_destroy(void* graph) {
    std::lock_guard<std::mutex> lock(g_res_mutex);
    SW_RES_CHECK(cudaGraphDestroy((cudaGraph_t)graph), "cudaGraphDestroy");
    return 0;
}

}  // extern "C"
