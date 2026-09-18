// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_kernels_decl.cuh — extern "C" __global__ prototypes shared by the
// kernel translation units and the launcher TU (sw_api.cu). nvcc links
// device code across TUs automatically when all .cu files are compiled in
// one nvcc invocation.
// ---------------------------------------------------------------------------
#ifndef SW_KERNELS_DECL_CUH_
#define SW_KERNELS_DECL_CUH_

#pragma once

// --- primitive parity probes (sw_primitives.cu) -----------------------------
extern "C" __global__ void sw_probe_expf(const float* __restrict__ x,
                                         float* __restrict__ y, long long n);
extern "C" __global__ void sw_probe_logf(const float* __restrict__ x,
                                         float* __restrict__ y, long long n);
extern "C" __global__ void sw_probe_powf(const float* __restrict__ x,
                                         const float* __restrict__ e,
                                         float* __restrict__ y, long long n);
extern "C" __global__ void sw_probe_opmath_powf(const float* __restrict__ x,
                                                float* __restrict__ y,
                                                long long n, int e);
extern "C" __global__ void sw_probe_arith(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ add,
                                          float* __restrict__ sub,
                                          float* __restrict__ mul,
                                          float* __restrict__ dv, long long n);
extern "C" __global__ void sw_probe_contract(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             const float* __restrict__ c,
                                             float* __restrict__ y, long long n);
extern "C" __global__ void sw_probe_maximum(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ y, long long n);
extern "C" __global__ void sw_probe_narith(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ add,
                                           float* __restrict__ sub,
                                           float* __restrict__ mul,
                                           float* __restrict__ dv, long long n);
extern "C" __global__ void sw_probe_is_libm_lane(
    const long long* __restrict__ i_, const long long* __restrict__ n_,
    const int* __restrict__ T_, const int* __restrict__ grain_,
    int* __restrict__ y, long long cnt);

// --- march kernels (sw_march.cu) ---------------------------------------------
extern "C" __global__ void sw_march_svf_shadow(
    const float* __restrict__ a, const float* __restrict__ vegdem,
    const float* __restrict__ vegdem2, const int* __restrict__ dxs,
    const int* __restrict__ dys, const float* __restrict__ dzs,
    int rows, int cols, int count,
    float* __restrict__ sh_out, float* __restrict__ vegsh_out,
    float* __restrict__ vbsh_out);
extern "C" __global__ void sw_march_wallheight23(
    const float* __restrict__ a, const float* __restrict__ vegdem,
    const float* __restrict__ vegdem2, const int* __restrict__ dxs,
    const int* __restrict__ dys, const float* __restrict__ dzs,
    const float* __restrict__ dzprevs, int rows, int cols, int count,
    float* __restrict__ sh_out, float* __restrict__ vegsh_out,
    float* __restrict__ vbsh_out);

// --- SVF fold kernel (sw_fold.cu) -------------------------------------------
extern "C" __global__ void sw_fold_kernel(
    const unsigned char* __restrict__ veg_bytes,
    const unsigned char* __restrict__ vbsh_bytes,
    const float* __restrict__ vegdem2, const float* __restrict__ svf_building,
    const float* __restrict__ w_iso, const float* __restrict__ w_aniso,
    const int* __restrict__ ring, const int* __restrict__ na,
    const signed char* __restrict__ dir_e, const signed char* __restrict__ dir_s,
    const signed char* __restrict__ dir_w, const signed char* __restrict__ dir_n,
    const unsigned char* __restrict__ mask, float* __restrict__ out,
    float last_const, float one_minus_trans, int rows, int cols, int do_mask);

// --- radiation kernels (sw_radiation.cu) -------------------------------------
extern "C" __global__ void sw_rad_day_kernel(
    const float* __restrict__ buildings, const float* __restrict__ aspect,
    const float* __restrict__ wallbol, const float* __restrict__ alb_grid,
    const float* __restrict__ emis_grid, const float* __restrict__ svfbuveg,
    const float* __restrict__ diffsh,
    const unsigned char* __restrict__ sh_pb,
    const unsigned char* __restrict__ veg_pb,
    const unsigned char* __restrict__ vbsh_pb,
    const unsigned char* __restrict__ sun_pb,
    const unsigned char* __restrict__ shd_pb,
    const long long* __restrict__ dp_rank,
    const signed char* __restrict__ guard_true,
    const float* __restrict__ shadow, const float* __restrict__ sunwall,
    const float* __restrict__ albshadow, const float* __restrict__ alb,
    const float* __restrict__ Lup_pre, const float* __restrict__ gvflup_extra,
    const float* __restrict__ lv2, const float* __restrict__ ster,
    const float* __restrict__ psin, const float* __restrict__ pcos,
    const float* __restrict__ lumChi, const float* __restrict__ lsky_d2,
    const float* __restrict__ lsky_s2,
    const signed char* __restrict__ card_e, const signed char* __restrict__ card_s,
    const signed char* __restrict__ card_w, const signed char* __restrict__ card_n,
    const float* __restrict__ ccos_e, const float* __restrict__ ccos_s,
    const float* __restrict__ ccos_w, const float* __restrict__ ccos_n,
    const float* __restrict__ walk_az_low, const float* __restrict__ walk_az_high,
    const int* __restrict__ walk_az_branch, const int* __restrict__ walk_dy,
    const int* __restrict__ walk_dx,
    const signed char* __restrict__ jE, const signed char* __restrict__ jS,
    const signed char* __restrict__ jW, const signed char* __restrict__ jN,
    const float* __restrict__ F_sh, const float* __restrict__ Tg_plane,
    const float* __restrict__ m_lup_in, const float* __restrict__ m_e_in,
    const float* __restrict__ m_s_in, const float* __restrict__ m_w_in,
    const float* __restrict__ m_n_in, const float* __restrict__ m_tg_in,
    float* __restrict__ o_tmrt, float* __restrict__ o_kdown,
    float* __restrict__ o_kup, float* __restrict__ o_ldown,
    float* __restrict__ o_lup, float* __restrict__ o_ke, float* __restrict__ o_ks,
    float* __restrict__ o_kw, float* __restrict__ o_kn, float* __restrict__ o_le,
    float* __restrict__ o_ls, float* __restrict__ o_lw, float* __restrict__ o_ln,
    float* __restrict__ o_ksidei, float* __restrict__ o_tgout,
    float* __restrict__ o_lside, float* __restrict__ o_ksided,
    float* __restrict__ o_drad, float* __restrict__ o_kside,
    float* __restrict__ n_lup, float* __restrict__ n_e, float* __restrict__ n_s,
    float* __restrict__ n_w, float* __restrict__ n_n, float* __restrict__ n_tg,
    float ks_sun, float ks_shd, float radI, float radD, float radG,
    float sinalt, float cosalt, double veg64, double shd64, double sun64,
    float ta273, float Lwall32, float Ta32,
    float w1_0, float w1_1, float w1_2, float w1_3, float w1_4, float w1_5,
    int rows, int cols, int n_patches, int kside_n, int fd_eq_1,
    int branch2, int sun_stride);
extern "C" __global__ void sw_rad_night_kernel(
    const unsigned char* __restrict__ sh_pb,
    const unsigned char* __restrict__ veg_pb,
    const unsigned char* __restrict__ vbsh_pb,
    const float* __restrict__ night_Lup,
    const float* __restrict__ ster, const float* __restrict__ psin,
    const float* __restrict__ pcos, const float* __restrict__ lsky_d2,
    const float* __restrict__ lsky_s2,
    const signed char* __restrict__ card_e, const signed char* __restrict__ card_s,
    const signed char* __restrict__ card_w, const signed char* __restrict__ card_n,
    const float* __restrict__ ccos_e, const float* __restrict__ ccos_s,
    const float* __restrict__ ccos_w, const float* __restrict__ ccos_n,
    double veg64, double shd64,
    float* __restrict__ o_tmrt, float* __restrict__ o_ldown,
    float* __restrict__ o_lside, float* __restrict__ o_le,
    float* __restrict__ o_ls, float* __restrict__ o_lw,
    float* __restrict__ o_ln,
    int rows, int cols, int n_patches);

// --- UTCI kernels (sw_utci.cu) -----------------------------------------------
extern "C" __global__ void sw_utci_count_kernel(
    const float* __restrict__ ta, const float* __restrict__ rh,
    const float* __restrict__ tmrt, const float* __restrict__ va,
    long long* __restrict__ counts, int rows, int cols);
extern "C" __global__ void sw_utci_fill_kernel(
    const float* __restrict__ ta, const float* __restrict__ rh,
    const float* __restrict__ tmrt, const float* __restrict__ va,
    const long long* __restrict__ offs, float* __restrict__ out,
    long long n, int T, int grain, int rows, int cols);
extern "C" __global__ void sw_utci_sparse_kernel(
    const float* __restrict__ ta, const float* __restrict__ rh,
    const float* __restrict__ tmrt, const float* __restrict__ va,
    float* __restrict__ out, long long n, int T, int grain);
extern "C" __global__ void sw_utci_sparse_bucket_kernel(
    const float* __restrict__ ta, const float* __restrict__ rh,
    const float* __restrict__ tmrt, const float* __restrict__ va,
    float* __restrict__ out, const long long* __restrict__ n_dev,
    int T, int grain);

#endif  // SW_KERNELS_DECL_CUH_
