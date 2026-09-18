// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_api.cu — torch-free C ABI over the CUDA runtime (TASKS T12).
//
// The Python host (native/cuda/host.py) loads the built shared object via
// ctypes. Every entry point returns 0 on success and a non-zero CUDA error
// code on failure; sw_last_error() carries the formatted message. All
// device memory is allocated through this ABI so buffer ownership is
// explicit and lives in exactly one place (the host.py DeviceArray).
//
// The build variant (strict canonical vs mutation) is baked in at compile
// time and reported by sw_build_flags() — the host asserts the expected
// variant string before running any canonical-parity gate.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#include "../include/sw_strict_math.cuh"
#include "../include/sw_kernels_decl.cuh"

static char g_last_error[1024] = "no error";
static std::mutex g_cuda_mutex;

extern "C" {

const char* sw_last_error() { return g_last_error; }

// build fingerprint: flags actually used for this object (build.py passes
// -DSW_BUILD_FLAG_STRING="...")
#ifndef SW_BUILD_FLAG_STRING
#define SW_BUILD_FLAG_STRING "unspecified"
#endif
const char* sw_build_flags() { return SW_BUILD_FLAG_STRING; }
int sw_abi_version() { return 1; }

int sw_device_count() {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "cudaGetDeviceCount: %s",
                 cudaGetErrorString(err));
        return -1;
    }
    return n;
}

int sw_device_name(int ordinal, char* buf, int buflen) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, ordinal);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "cudaGetDeviceProperties: %s",
                 cudaGetErrorString(err));
        return (int)err;
    }
    snprintf(buf, buflen, "%s (sm_%d%d, %d MB)", prop.name, prop.major,
             prop.minor, (int)(prop.totalGlobalMem >> 20));
    return 0;
}

// --- device memory ---------------------------------------------------------
// T13: monotonic allocation accounting — every cudaMalloc through this
// ABI bumps the counters (never decremented). The dispatch gate asserts
// a CPU-selected request makes ZERO new allocations, and the residency
// stress gate asserts steady-state cycles allocate nothing (leak fence).
static long long g_sw_alloc_count = 0;
static long long g_sw_alloc_bytes = 0;

int sw_alloc(void** out, long long bytes) {
    cudaError_t err = cudaMalloc(out, (size_t)bytes);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "cudaMalloc(%lld): %s",
                 bytes, cudaGetErrorString(err));
        return (int)err;
    }
    g_sw_alloc_count += 1;
    g_sw_alloc_bytes += bytes;
    return 0;
}

int sw_alloc_count() { return (int)g_sw_alloc_count; }

long long sw_alloc_bytes() { return g_sw_alloc_bytes; }

int sw_free(void* p) {
    cudaError_t err = cudaFree(p);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "cudaFree: %s",
                 cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int sw_copy_h2d(const void* host, void* dev, long long bytes) {
    cudaError_t err = cudaMemcpy(dev, host, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "MemcpyH2D(%lld): %s", bytes,
                 cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int sw_copy_d2h(const void* dev, void* host, long long bytes) {
    cudaError_t err = cudaMemcpy(host, dev, (size_t)bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "MemcpyD2H(%lld): %s", bytes,
                 cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int sw_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof g_last_error, "DeviceSynchronize: %s",
                 cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

// --- CUDA-event timing (kernel-only measurement) ---------------------------
// One outstanding interval at a time — the host serialises benchmark runs.
static cudaEvent_t g_ev_start = nullptr, g_ev_stop = nullptr;

int sw_timer_begin() {
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    if (!g_ev_start) {
        if (cudaEventCreate(&g_ev_start) != cudaSuccess ||
            cudaEventCreate(&g_ev_stop) != cudaSuccess) {
            snprintf(g_last_error, sizeof g_last_error, "cudaEventCreate failed");
            return -1;
        }
    }
    return cudaEventRecord(g_ev_start, 0) == cudaSuccess ? 0 : -1;
}

float sw_timer_end() {  // ms elapsed since the matching sw_timer_begin
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    if (!g_ev_start) return -1.0f;
    if (cudaEventRecord(g_ev_stop, 0) != cudaSuccess) return -1.0f;
    if (cudaEventSynchronize(g_ev_stop) != cudaSuccess) return -1.0f;
    float ms = -1.0f;
    cudaEventElapsedTime(&ms, g_ev_start, g_ev_stop);
    return ms;
}

// --- launch helpers --------------------------------------------------------
#define SW_LAUNCH_1D(kernel, n, ...)                                            \
    do {                                                                       \
        std::lock_guard<std::mutex> lock(g_cuda_mutex);                        \
        long long blocks = ((n) + 255) / 256;                                  \
        if (blocks < 1) blocks = 1;                                            \
        if (blocks > (1LL << 31) - 1) {                                        \
            snprintf(g_last_error, sizeof g_last_error, "grid too large");     \
            return -1;                                                          \
        }                                                                      \
        kernel<<<(unsigned)blocks, 256>>>(__VA_ARGS__);                        \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            snprintf(g_last_error, sizeof g_last_error, "launch %s: %s",       \
                     #kernel, cudaGetErrorString(err));                        \
            return (int)err;                                                   \
        }                                                                      \
        return 0;                                                              \
    } while (0)

// --- primitive parity probes ------------------------------------------------
int sw_run_probe_expf(const float* x, float* y, long long n) {
    SW_LAUNCH_1D(sw_probe_expf, n, x, y, n);
}
int sw_run_probe_logf(const float* x, float* y, long long n) {
    SW_LAUNCH_1D(sw_probe_logf, n, x, y, n);
}
int sw_run_probe_powf(const float* x, const float* e, float* y, long long n) {
    SW_LAUNCH_1D(sw_probe_powf, n, x, e, y, n);
}
int sw_run_probe_opmath_powf(const float* x, float* y, long long n, int e) {
    SW_LAUNCH_1D(sw_probe_opmath_powf, n, x, y, n, e);
}
int sw_run_probe_arith(const float* a, const float* b, float* add, float* sub,
                       float* mul, float* dv, long long n) {
    SW_LAUNCH_1D(sw_probe_arith, n, a, b, add, sub, mul, dv, n);
}
int sw_run_probe_contract(const float* a, const float* b, const float* c,
                          float* y, long long n) {
    SW_LAUNCH_1D(sw_probe_contract, n, a, b, c, y, n);
}
int sw_run_probe_maximum(const float* a, const float* b, float* y, long long n) {
    SW_LAUNCH_1D(sw_probe_maximum, n, a, b, y, n);
}
int sw_run_probe_narith(const float* a, const float* b, float* add, float* sub,
                        float* mul, float* dv, long long n) {
    SW_LAUNCH_1D(sw_probe_narith, n, a, b, add, sub, mul, dv, n);
}
int sw_run_probe_is_libm_lane(const long long* i, const long long* n,
                              const int* T, const int* grain, int* y,
                              long long cnt) {
    SW_LAUNCH_1D(sw_probe_is_libm_lane, cnt, i, n, T, grain, y, cnt);
}

// --- march (sw_march.cu) ------------------------------------------------------
// ABI convention: all pointers first, scalars last (matches host.py).
int sw_run_march_svf_shadow(const float* a, const float* vegdem,
                            const float* vegdem2, const int* dxs,
                            const int* dys, const float* dzs,
                            float* sh_out, float* vegsh_out,
                            float* vbsh_out, int rows, int cols,
                            int count) {
    long long n = (long long)rows * cols;
    SW_LAUNCH_1D(sw_march_svf_shadow, n, a, vegdem, vegdem2, dxs, dys, dzs,
                 rows, cols, count, sh_out, vegsh_out, vbsh_out);
}
int sw_run_march_wallheight23(const float* a, const float* vegdem,
                              const float* vegdem2, const int* dxs,
                              const int* dys, const float* dzs,
                              const float* dzprevs, float* sh_out,
                              float* vegsh_out, float* vbsh_out, int rows,
                              int cols, int count) {
    long long n = (long long)rows * cols;
    SW_LAUNCH_1D(sw_march_wallheight23, n, a, vegdem, vegdem2, dxs, dys, dzs,
                 dzprevs, rows, cols, count, sh_out, vegsh_out, vbsh_out);
}

// --- fold (sw_fold.cu) -------------------------------------------------------
// ABI convention: all pointers first, scalars last (matches host.py). The
// masked route passes mask == NULL with do_mask == 0; with do_mask == 1 the
// kernel leaves every unmasked cell's outputs at the caller's pre-filled
// base bits (affected-chunk-only contract).
int sw_run_fold(const unsigned char* veg, const unsigned char* vbsh,
                const float* vegdem2, const float* svf_building,
                const float* w_iso, const float* w_aniso, const int* ring,
                const int* na, const signed char* dir_e,
                const signed char* dir_s, const signed char* dir_w,
                const signed char* dir_n, const unsigned char* mask,
                float* out, float last_const, float one_minus_trans, int rows,
                int cols, int do_mask) {
    long long n = (long long)rows * cols;
    SW_LAUNCH_1D(sw_fold_kernel, n, veg, vbsh, vegdem2, svf_building, w_iso,
                 w_aniso, ring, na, dir_e, dir_s, dir_w, dir_n, mask, out,
                 last_const, one_minus_trans, rows, cols, do_mask);
}

// --- radiation (sw_radiation.cu) ---------------------------------------------
// ABI convention: all pointers first, scalars last (matches host.py). The
// day kernel's 50 pointer args + 19 scalars mirror the numba fused-day
// signature exactly.
int sw_run_rad_day(
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
    int sun_stride) {
    long long n = (long long)rows * cols;
    SW_LAUNCH_1D(sw_rad_day_kernel, n,
                 buildings, aspect, wallbol, alb_grid, emis_grid, svfbuveg,
                 diffsh, sh_pb, veg_pb, vbsh_pb, sun_pb, shd_pb, dp_rank,
                 guard_true, shadow, sunwall, albshadow, alb, Lup_pre,
                 gvflup_extra, lv2, ster, psin, pcos, lumChi, lsky_d2, lsky_s2,
                 card_e, card_s, card_w, card_n, ccos_e, ccos_s, ccos_w,
                 ccos_n, walk_az_low, walk_az_high, walk_az_branch, walk_dy,
                 walk_dx, jE, jS, jW, jN, F_sh, Tg_plane, m_lup_in, m_e_in,
                 m_s_in, m_w_in, m_n_in, m_tg_in, o_tmrt, o_kdown, o_kup,
                 o_ldown, o_lup, o_ke, o_ks, o_kw, o_kn, o_le, o_ls, o_lw,
                 o_ln, o_ksidei, o_tgout, o_lside, o_ksided, o_drad, o_kside,
                 n_lup, n_e, n_s, n_w, n_n, n_tg, ks_sun, ks_shd, radI, radD,
                 radG, sinalt, cosalt, veg64, shd64, sun64, ta273, Lwall32,
                 Ta32, w1_0, w1_1, w1_2, w1_3, w1_4, w1_5, rows, cols,
                 n_patches, kside_n, fd_eq_1, branch2, sun_stride);
}

int sw_run_rad_night(
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
    int rows, int cols, int n_patches) {
    long long n = (long long)rows * cols;
    SW_LAUNCH_1D(sw_rad_night_kernel, n,
                 sh_pb, veg_pb, vbsh_pb, night_Lup, ster, psin, pcos, lsky_d2,
                 lsky_s2, card_e, card_s, card_w, card_n, ccos_e, ccos_s,
                 ccos_w, ccos_n, veg64, shd64, o_tmrt, o_ldown, o_lside, o_le,
                 o_ls, o_lw, o_ln, rows, cols, n_patches);
}

// --- UTCI (sw_utci.cu) --------------------------------------------------------
// ABI convention: all pointers first, scalars last (matches host.py). The
// dense route is two launches around a HOST prefix sum (offs has rows+1
// entries, offs[0]=0, exactly the CPU kernel's offsets array); ``out``
// arrives pre-filled with -999 and only valid lanes are written.
int sw_utci_count(const float* ta, const float* rh, const float* tmrt,
                  const float* va, long long* counts, int rows, int cols) {
    SW_LAUNCH_1D(sw_utci_count_kernel, rows, ta, rh, tmrt, va, counts,
                 rows, cols);
}
int sw_utci_fill(const float* ta, const float* rh, const float* tmrt,
                 const float* va, const long long* offs, float* out,
                 long long n, int T, int grain, int rows, int cols) {
    long long cells = (long long)rows * cols;
    SW_LAUNCH_1D(sw_utci_fill_kernel, cells, ta, rh, tmrt, va, offs, out,
                 n, T, grain, rows, cols);
}
int sw_utci_sparse(const float* ta, const float* rh, const float* tmrt,
                   const float* va, float* out, long long n, int T,
                   int grain) {
    SW_LAUNCH_1D(sw_utci_sparse_kernel, n, ta, rh, tmrt, va, out, n, T,
                 grain);
}

}  // extern "C"
