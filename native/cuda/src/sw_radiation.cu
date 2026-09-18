// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_radiation.cu — CUDA radiation stage (ultrafast bitwise plan, TASKS
// T12; DESIGN 9.3 ordered fusion). One thread per target cell replays, IN
// EXECUTION ORDER, the exact primitive graph of solweig_core/numba_cpu/
// radiation.py's fused kernels (_fused_day_kernel / _fused_night_kernel —
// the frozen T08 reference, bit-equal to the original Solweig_2022a_calc
// radiation chain for the site_500 configuration).
//
// Preserved semantics (every line of the numba kernels):
//   * stage order per cell: dRad -> 18-azimuth sunonsurface walk (11
//     steps, stale-border recursion) -> TsWaveDelay x6 -> Kup family ->
//     Kside -> Kdown -> define_patch two-pass with the CELL-LOCAL
//     reflection barrier -> Lside_veg -> Sstr/Tmrt -> POI cardinals;
//   * every float32 operand chain keeps the source association (no
//     reassociation, no FMA contraction — the strict build guarantees
//     separate mul/add; divisions are IEEE via --prec-div=true, sqrt via
//     --prec-sqrt=true);
//   * the veg/sun/shaded define_patch terms are F64 chains with ONE
//     rounding per accumulate ((float)((double)acc + P * mask)) exactly
//     as the torch `f32 += f64-plane` promotion mirror;
//   * frozen bundle scalars/tables arrive as DATA (captured bits — never
//     recomputed on device);
//   * the walk's temp locals keep the PREVIOUS step's value outside the
//     write window (zeros before step 0) — the stale-border semantics of
//     the source's shifted-slice writes;
//   * no cross-thread communication: a lane's outputs depend only on the
//     cell's read extent (target cell + walk offsets + packed cubes).
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include "../include/sw_kernels_decl.cuh"
#include "../include/sw_strict_math.cuh"

// patch bit i of a packed cube (rows, cols, stride) uint8, little order
__device__ __forceinline__ bool sw_pbit(const unsigned char *__restrict__ pb,
                                        long long cell, int stride, int i) {
    return ((pb[cell * stride + (i >> 3)] >> (i & 7)) & 1) != 0;
}

// TsWaveDelay blend: fl32(src*(1-w1)) + fl32(m_in*w1), rounded once each
__device__ __forceinline__ float sw_tsw_blend(float src, float w1,
                                              float m_in) {
    float omw = 1.0f - w1;
    return src * omw + m_in * w1;
}

extern "C" __global__ void sw_rad_day_kernel(
    const float *__restrict__ buildings, const float *__restrict__ aspect,
    const float *__restrict__ wallbol, const float *__restrict__ alb_grid,
    const float *__restrict__ emis_grid, const float *__restrict__ svfbuveg,
    const float *__restrict__ diffsh,
    const unsigned char *__restrict__ sh_pb,
    const unsigned char *__restrict__ veg_pb,
    const unsigned char *__restrict__ vbsh_pb,
    const unsigned char *__restrict__ sun_pb,
    const unsigned char *__restrict__ shd_pb,
    const long long *__restrict__ dp_rank,
    const signed char *__restrict__ guard_true,
    const float *__restrict__ shadow, const float *__restrict__ sunwall,
    const float *__restrict__ albshadow, const float *__restrict__ alb,
    const float *__restrict__ Lup_pre, const float *__restrict__ gvflup_extra,
    const float *__restrict__ lv2, const float *__restrict__ ster,
    const float *__restrict__ psin, const float *__restrict__ pcos,
    const float *__restrict__ lumChi, const float *__restrict__ lsky_d2,
    const float *__restrict__ lsky_s2,
    const signed char *__restrict__ card_e, const signed char *__restrict__ card_s,
    const signed char *__restrict__ card_w, const signed char *__restrict__ card_n,
    const float *__restrict__ ccos_e, const float *__restrict__ ccos_s,
    const float *__restrict__ ccos_w, const float *__restrict__ ccos_n,
    const float *__restrict__ walk_az_low, const float *__restrict__ walk_az_high,
    const int *__restrict__ walk_az_branch, const int *__restrict__ walk_dy,
    const int *__restrict__ walk_dx,
    const signed char *__restrict__ jE, const signed char *__restrict__ jS,
    const signed char *__restrict__ jW, const signed char *__restrict__ jN,
    const float *__restrict__ F_sh, const float *__restrict__ Tg_plane,
    const float *__restrict__ m_lup_in, const float *__restrict__ m_e_in,
    const float *__restrict__ m_s_in, const float *__restrict__ m_w_in,
    const float *__restrict__ m_n_in, const float *__restrict__ m_tg_in,
    float *__restrict__ o_tmrt, float *__restrict__ o_kdown,
    float *__restrict__ o_kup, float *__restrict__ o_ldown,
    float *__restrict__ o_lup, float *__restrict__ o_ke, float *__restrict__ o_ks,
    float *__restrict__ o_kw, float *__restrict__ o_kn, float *__restrict__ o_le,
    float *__restrict__ o_ls, float *__restrict__ o_lw, float *__restrict__ o_ln,
    float *__restrict__ o_ksidei, float *__restrict__ o_tgout,
    float *__restrict__ o_lside, float *__restrict__ o_ksided,
    float *__restrict__ o_drad, float *__restrict__ o_kside,
    float *__restrict__ n_lup, float *__restrict__ n_e, float *__restrict__ n_s,
    float *__restrict__ n_w, float *__restrict__ n_n, float *__restrict__ n_tg,
    float ks_sun, float ks_shd, float radI, float radD, float radG,
    float sinalt, float cosalt, double veg64, double shd64, double sun64,
    float ta273, float Lwall32, float Ta32,
    float w1_0, float w1_1, float w1_2, float w1_3, float w1_4, float w1_5,
    int rows, int cols, int n_patches, int kside_n, int fd_eq_1,
    int branch2, int sun_stride) {

    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) {
        return;
    }
    const float one = 1.0f;
    const float half = 0.5f;
    const float m_one = -1.0f;
    const float alb_b = 0.2f;
    const float second32 = 11.0f;
    const float first32 = 1.0f;
    const float first_p1 = 2.0f;
    const float second_p1 = 12.0f;
    const float c05 = 0.5f;
    const float c04 = 0.4f;
    const float c09 = 0.9f;
    const float f_cyl = 0.28f;
    const float f_up = 0.06f;
    const float f_side = 0.22f;
    const float abs_k = 0.7f;
    const float abs_l = 0.95f;
    const float sbc = 5.67051e-8f;
    const float pi32 = 3.14159274101257324f;  // fl32(pi)
    const float ome = one - 0.9f;             // fl32(1.0) - fl32(0.9)
    const float n18 = 18.0f;
    const float n9 = 9.0f;
    const float r2732 = 273.2f;
    const float zero = 0.0f;
    int r = (int)(cell / cols);
    int c = (int)(cell % cols);

    // ---- dRad (aniLum over patches, ascending idx) ----
    float ani = 0.0f;
    for (int idx = 0; idx < n_patches; ++idx) {
        ani = ani + diffsh[(cell * n_patches + idx)] * lv2[idx];
    }
    float dRad = ani * radD;
    o_drad[cell] = dRad;

    // ---- 18-azimuth sunonsurface walk (per-cell stale-border recursion) ----
    float acc_lup = 0.0f, acc_alb = 0.0f, acc_nosh = 0.0f;
    float ce_l = 0.0f, ce_a = 0.0f, ce_n = 0.0f;
    float cs_l = 0.0f, cs_a = 0.0f, cs_n = 0.0f;
    float cw_l = 0.0f, cw_a = 0.0f, cw_n = 0.0f;
    float cn_l = 0.0f, cn_a = 0.0f, cn_n = 0.0f;
    float b_row = buildings[cell];
    float a_row = alb_grid[cell];
    float sh_rc = shadow[cell];
    float no_b_term = a_row * (b_row * m_one + one);

    for (int j = 0; j < 18; ++j) {
        float f = b_row;
        float tbu = 0.0f, tsh_p = 0.0f, tlsh = 0.0f, talsh = 0.0f;
        float tanosh = 0.0f, tws = 0.0f;
        float wsh = 0.0f, wwall = 0.0f, wlupsh = 0.0f, wlwall = 0.0f;
        float walbsh = 0.0f, walbwall = 0.0f, walbnosh = 0.0f;
        float walbwnosh = 0.0f, tbub = 0.0f, tbubw = 0.0f;
        float wsh_f = 0.0f, wwall_f = 0.0f, wlup_f = 0.0f, wlwall_f = 0.0f;
        float walb_f = 0.0f, walbw_f = 0.0f, walbn_f = 0.0f, walbwn_f = 0.0f;

        for (int n = 0; n < 11; ++n) {
            int dy = walk_dy[j * 11 + n];
            int dx = walk_dx[j * 11 + n];
            int adx = dx >= 0 ? dx : -dx;
            int ady = dy >= 0 ? dy : -dy;
            int xp1 = -((dx - adx) / 2);
            int xp2 = rows - (dx + adx) / 2;
            int yp1 = -((dy - ady) / 2);
            int yp2 = cols - (dy + ady) / 2;
            if (xp1 <= r && r < xp2 && yp1 <= c && c < yp2) {
                long long src = (long long)(r + dx) * cols + (c + dy);
                tbu = buildings[src];
                tsh_p = shadow[src];
                tlsh = Lup_pre[src];
                talsh = albshadow[src];
                tanosh = alb[src];
                tws = sunwall[src];
            }
            if (tbu < f) {
                f = tbu;
            }
            wsh = wsh + tsh_p * f;
            wlupsh = wlupsh + tlsh * f;
            walbsh = walbsh + talsh * f;
            walbnosh = walbnosh + tanosh * f;
            float tempb = tws * f;
            float tempbwall = f * m_one + one;
            tbub = (tempb + tbub) > zero ? one : zero;
            tbubw = (tempbwall + tbubw) > zero ? one : zero;
            wlwall = wlwall + tbub * Lwall32;
            walbwall = walbwall + tbub * alb_b;
            wwall = wwall + tbub;
            walbwnosh = walbwnosh + tbubw * alb_b;
            if (n == 0) {  // n+1 <= first (== 1): step-0 snapshots
                wsh_f = wsh / one;
                wwall_f = wwall / one;
                wlup_f = wlupsh / one;
                wlwall_f = wlwall / one;
                walb_f = walbsh / one;
                walbw_f = walbwall / one;
                walbn_f = walbnosh / one;
                walbwn_f = walbwnosh / one;
            }
        }

        float wif = wwall_f > zero ? one : zero;
        float wis = wwall > zero ? one : zero;
        float lif = walbwn_f > zero ? one : zero;
        float lis_ = walbwnosh > zero ? one : zero;
        float nif_ = wif * m_one + one;
        float nis_ = wis * m_one + one;
        float nlf_ = lif * m_one + one;
        float nls_ = lis_ * m_one + one;

        int br = walk_az_branch[j];
        float lo = walk_az_low[j];
        float hi = walk_az_high[j];
        float asp = aspect[cell];
        float facesh;
        if (br == 1) {
            float b2 = (asp < lo) || (asp >= hi) ? one : zero;
            facesh = b2 - wallbol[cell] + one;
        } else {
            float b2 = (asp > lo) || (asp <= hi) ? one : zero;
            facesh = b2 * m_one + one;
        }

        float keep = (wwall == second32 ? one : zero) - facesh;
        if (keep == m_one) {
            keep = zero;
        }
        if (keep == one) {
            wwall = zero;
            wlwall = zero;
            walbwall = zero;
        }

        float gvf2 = ((wwall + wsh) / second_p1) * wis +
                     (wsh / second32) * nis_;
        if (gvf2 > one) {
            gvf2 = one;
        }
        float gl1 = ((wlwall_f + wlup_f) / first_p1) * wif +
                    (wlup_f / first32) * nif_;
        float gl2 = ((wlwall + wlupsh) / second_p1) * wis +
                    (wlupsh / second32) * nis_;
        float ga1 = ((walbw_f + walb_f) / first_p1) * wif +
                    (walb_f / first32) * nif_;
        float ga2 = ((walbwall + walbsh) / second_p1) * wis +
                    (walbsh / second32) * nis_;
        float gn1 = ((walbwn_f + walbn_f) / first_p1) * lif +
                    (walbn_f / first32) * nlf_;
        // gvfalbnosh2 divides by `second` (NOT second+1) — source
        // asymmetry kept verbatim
        float gn2 = ((walbwnosh + walbnosh) / second32) * lis_ +
                    (walbnosh / second32) * nls_;

        float g_lup = (gl1 * c05 + gl2 * c04) / c09 + gvflup_extra[cell];
        float g_alb = (ga1 * c05 + ga2 * c04) / c09 + no_b_term * sh_rc;
        float g_nosh = (gn1 * c05 + gn2 * c04) / c09 * b_row + no_b_term;

        acc_lup = acc_lup + g_lup;
        acc_alb = acc_alb + g_alb;
        acc_nosh = acc_nosh + g_nosh;
        if (jE[j]) {
            ce_l = ce_l + g_lup; ce_a = ce_a + g_alb; ce_n = ce_n + g_nosh;
        }
        if (jS[j]) {
            cs_l = cs_l + g_lup; cs_a = cs_a + g_alb; cs_n = cs_n + g_nosh;
        }
        if (jW[j]) {
            cw_l = cw_l + g_lup; cw_a = cw_a + g_alb; cw_n = cw_n + g_nosh;
        }
        if (jN[j]) {
            cn_l = cn_l + g_lup; cn_a = cn_a + g_alb; cn_n = cn_n + g_nosh;
        }
    }

    float ta_pow4 = (sbc * emis_grid[cell]) * ta273;
    float gvfLup = acc_lup / n18 + ta_pow4;
    float gvfLupE = ce_l / n9 + ta_pow4;
    float gvfLupS = cs_l / n9 + ta_pow4;
    float gvfLupW = cw_l / n9 + ta_pow4;
    float gvfLupN = cn_l / n9 + ta_pow4;
    float gvfalb = acc_alb / n18;
    float gvfalbE = ce_a / n9;
    float gvfalbS = cs_a / n9;
    float gvfalbW = cw_a / n9;
    float gvfalbN = cn_a / n9;
    float gvfNosh = acc_nosh / n18;
    float gvfNoshE = ce_n / n9;
    float gvfNoshS = cs_n / n9;
    float gvfNoshW = cw_n / n9;
    float gvfNoshN = cn_n / n9;

    // ---- TsWaveDelay x6 (entry fd/timeadd uniform per call) ----
    float m_in = fd_eq_1 ? gvfLup : m_lup_in[cell];
    float lup_out = sw_tsw_blend(gvfLup, w1_0, m_in);
    o_lup[cell] = lup_out;
    n_lup[cell] = branch2 ? lup_out : m_in;
    m_in = fd_eq_1 ? gvfLupE : m_e_in[cell];
    float lup_e = sw_tsw_blend(gvfLupE, w1_1, m_in);
    n_e[cell] = branch2 ? lup_e : m_in;
    m_in = fd_eq_1 ? gvfLupS : m_s_in[cell];
    float lup_s = sw_tsw_blend(gvfLupS, w1_2, m_in);
    n_s[cell] = branch2 ? lup_s : m_in;
    m_in = fd_eq_1 ? gvfLupW : m_w_in[cell];
    float lup_w = sw_tsw_blend(gvfLupW, w1_3, m_in);
    n_w[cell] = branch2 ? lup_w : m_in;
    m_in = fd_eq_1 ? gvfLupN : m_n_in[cell];
    float lup_n = sw_tsw_blend(gvfLupN, w1_4, m_in);
    n_n[cell] = branch2 ? lup_n : m_in;
    float TgTemp = Tg_plane[cell] * sh_rc + Ta32;
    m_in = fd_eq_1 ? TgTemp : m_tg_in[cell];
    float tg_blend = sw_tsw_blend(TgTemp, w1_5, m_in);
    o_tgout[cell] = tg_blend;
    n_tg[cell] = branch2 ? tg_blend : m_in;

    // ---- Kup family ----
    float svf = svfbuveg[cell];
    float fsh = F_sh[cell];
    float inner = radD * svf +
                  alb_b * (one - svf) * (radG * (one - fsh) + radD * fsh);
    float Kup = (gvfalb * radI * sinalt) + inner * gvfNosh;
    float KupE = (gvfalbE * radI * sinalt) + inner * gvfNoshE;
    float KupS = (gvfalbS * radI * sinalt) + inner * gvfNoshS;
    float KupW = (gvfalbW * radI * sinalt) + inner * gvfNoshW;
    float KupN = (gvfalbN * radI * sinalt) + inner * gvfNoshN;
    o_kup[cell] = Kup;
    o_ke[cell] = KupE * half;
    o_ks[cell] = KupS * half;
    o_kw[cell] = KupW * half;
    o_kn[cell] = KupN * half;

    // ---- Kside (Kside's own sos bits 0..P-1) ----
    float KsideI = (sh_rc * radI) * cosalt;
    o_ksidei[cell] = KsideI;
    float KsideD = 0.0f;
    float Krs = 0.0f;
    float Krsh = 0.0f;
    float Krv = 0.0f;
    for (int idx = 0; idx < n_patches; ++idx) {
        KsideD = KsideD +
                 ((diffsh[cell * n_patches + idx] * lumChi[idx]) * pcos[idx]) *
                     ster[idx];
        bool vegb = sw_pbit(veg_pb, cell, (n_patches + 7) / 8, idx);
        bool vbshb = sw_pbit(vbsh_pb, cell, (n_patches + 7) / 8, idx);
        float tvb = ((!vegb) || (!vbshb)) ? one : zero;
        Krv = Krv + ((ks_shd * tvb) * ster[idx]) * pcos[idx];
        bool shb = sw_pbit(sh_pb, cell, (n_patches + 7) / 8, idx);
        float tsh_ = ((!shb) && vbshb) ? one : zero;
        bool slb = sw_pbit(sun_pb, cell, sun_stride, idx);
        bool sdb = sw_pbit(shd_pb, cell, sun_stride, idx);
        float slf = slb ? one : zero;
        float sdf = sdb ? one : zero;
        Krs = Krs + ((((ks_sun * slf) * tsh_) * ster[idx]) * pcos[idx]);
        Krsh = Krsh + ((((ks_shd * sdf) * tsh_) * ster[idx]) * pcos[idx]);
    }
    float Kside = ((((KsideI + KsideD) + Krs) + Krsh) + Krv);
    o_ksided[cell] = KsideD;
    o_kside[cell] = Kside;

    // ---- Kdown ----
    float Kdown = ((radI * sh_rc) * sinalt) + dRad +
                  (alb_b * (one - svf)) *
                      (radG * (one - fsh) + radD * fsh);
    o_kdown[cell] = Kdown;

    // ---- define_patch two-pass, cell-local reflection barrier ----
    float Ls_sky = 0.0f, Ld_sky = 0.0f;
    float Ls_veg = 0.0f, Ld_veg = 0.0f;
    float Ls_sun = 0.0f, Ld_sun = 0.0f;
    float Ls_sh = 0.0f, Ld_sh = 0.0f;
    float ae = 0.0f, aso = 0.0f, aw = 0.0f, an = 0.0f;
    int pstride = (n_patches + 7) / 8;
    for (int idx = 0; idx < n_patches; ++idx) {
        bool shb = sw_pbit(sh_pb, cell, pstride, idx);
        bool vegb = sw_pbit(veg_pb, cell, pstride, idx);
        bool vbshb = sw_pbit(vbsh_pb, cell, pstride, idx);
        float sky_m = (shb && vegb) ? one : zero;
        double veg_m = ((!vegb) || (!vbshb)) ? 1.0 : 0.0;
        float sun_m = ((!shb) && vbshb) ? one : zero;
        Ld_sky = Ld_sky + sky_m * lsky_d2[idx];
        Ls_sky = Ls_sky + sky_m * lsky_s2[idx];
        double st64 = (double)ster[idx];
        double pc64 = (double)pcos[idx];
        double ps64 = (double)psin[idx];
        double Pv_c = (veg64 * st64) * pc64;
        double Pv_s = (veg64 * st64) * ps64;
        Ls_veg = (float)((double)Ls_veg + Pv_c * veg_m);
        Ld_veg = (float)((double)Ld_veg + Pv_s * veg_m);
        bool ce_b = card_e[idx] != 0;
        bool cs_b = card_s[idx] != 0;
        bool cw_b = card_w[idx] != 0;
        bool cn_b = card_n[idx] != 0;
        if (ce_b) {
            ae = ae + (sky_m * lsky_s2[idx]) * ccos_e[idx];
            ae = (float)((double)ae + (Pv_c * (double)ccos_e[idx]) * veg_m);
        }
        if (cs_b) {
            aso = aso + (sky_m * lsky_s2[idx]) * ccos_s[idx];
            aso = (float)((double)aso + (Pv_c * (double)ccos_s[idx]) * veg_m);
        }
        if (cw_b) {
            aw = aw + (sky_m * lsky_s2[idx]) * ccos_w[idx];
            aw = (float)((double)aw + (Pv_c * (double)ccos_w[idx]) * veg_m);
        }
        if (cn_b) {
            an = an + (sky_m * lsky_s2[idx]) * ccos_n[idx];
            an = (float)((double)an + (Pv_c * (double)ccos_n[idx]) * veg_m);
        }

        double Psh_c = (shd64 * st64) * pc64;
        double Psh_s = (shd64 * st64) * ps64;
        double Psun_c = (sun64 * st64) * pc64;
        double Psun_s = (sun64 * st64) * ps64;
        if (guard_true[idx]) {
            long long k = (long long)kside_n + dp_rank[idx];
            bool sl = sw_pbit(sun_pb, cell, sun_stride, (int)k);
            bool sd = sw_pbit(shd_pb, cell, sun_stride, (int)k);
            double g1 = (sl && (sun_m == one)) ? 1.0 : 0.0;
            double g2 = (sd && (sun_m == one)) ? 1.0 : 0.0;
            Ls_sun = (float)((double)Ls_sun + Psun_c * g1);
            Ld_sun = (float)((double)Ld_sun + Psun_s * g1);
            Ls_sh = (float)((double)Ls_sh + Psh_c * g2);
            Ld_sh = (float)((double)Ld_sh + Psh_s * g2);
            if (ce_b) {
                ae = (float)((double)ae + (Psun_c * (double)ccos_e[idx]) * g1);
                ae = (float)((double)ae + (Psh_c * (double)ccos_e[idx]) * g2);
            }
            if (cs_b) {
                aso = (float)((double)aso + (Psun_c * (double)ccos_s[idx]) * g1);
                aso = (float)((double)aso + (Psh_c * (double)ccos_s[idx]) * g2);
            }
            if (cw_b) {
                aw = (float)((double)aw + (Psun_c * (double)ccos_w[idx]) * g1);
                aw = (float)((double)aw + (Psh_c * (double)ccos_w[idx]) * g2);
            }
            if (cn_b) {
                an = (float)((double)an + (Psun_c * (double)ccos_n[idx]) * g1);
                an = (float)((double)an + (Psh_c * (double)ccos_n[idx]) * g2);
            }
        } else {
            double sm64 = (double)sun_m;
            Ls_sh = (float)((double)Ls_sh + Psh_c * sm64);
            Ld_sh = (float)((double)Ld_sh + Psh_s * sm64);
            if (ce_b) {
                ae = (float)((double)ae + Psh_c * (double)ccos_e[idx] * sm64);
            }
            if (cs_b) {
                aso = (float)((double)aso + Psh_c * (double)ccos_s[idx] * sm64);
            }
            if (cw_b) {
                aw = (float)((double)aw + Psh_c * (double)ccos_w[idx] * sm64);
            }
            if (cn_b) {
                an = (float)((double)an + Psh_c * (double)ccos_n[idx] * sm64);
            }
        }
    }

    // barrier: reflected term needs only THIS cell's Ldown_sky
    float refl = (((Ld_sky + lup_out) * ome) * half) / pi32;
    float Ls_ref = 0.0f;
    float Ld_ref = 0.0f;
    for (int idx = 0; idx < n_patches; ++idx) {
        bool shb = sw_pbit(sh_pb, cell, pstride, idx);
        bool vegb = sw_pbit(veg_pb, cell, pstride, idx);
        bool vbshb = sw_pbit(vbsh_pb, cell, pstride, idx);
        float rm = ((!shb) || (!vegb) || (!vbshb)) ? one : zero;
        Ls_ref = Ls_ref + (((refl * ster[idx]) * pcos[idx]) * rm);
        Ld_ref = Ld_ref + (((refl * ster[idx]) * psin[idx]) * rm);
        if (card_e[idx]) {
            ae = ae + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_e[idx]);
        }
        if (card_s[idx]) {
            aso = aso + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_s[idx]);
        }
        if (card_w[idx]) {
            aw = aw + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_w[idx]);
        }
        if (card_n[idx]) {
            an = an + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_n[idx]);
        }
    }

    float Lside = ((((Ls_sky + Ls_veg) + Ls_sh) + Ls_sun) + Ls_ref);
    float Ldown = ((((Ld_sky + Ld_veg) + Ld_sh) + Ld_sun) + Ld_ref);
    o_lside[cell] = Lside;
    o_ldown[cell] = Ldown;

    // ---- Lside_veg (tsw cardinals * 0.5) ----
    float le_lv = lup_e * half;
    float ls_lv = lup_s * half;
    float lw_lv = lup_w * half;
    float ln_lv = lup_n * half;

    // ---- Sstr / Tmrt (cyl + anisotropic) ----
    float ke_s = KupE * half;
    float ks_s = KupS * half;
    float kw_s = KupW * half;
    float kn_s = KupN * half;
    float short_ = Kside * f_cyl + (Kdown + Kup) * f_up +
                   (((kn_s + ke_s) + ks_s) + kw_s) * f_side;
    float long_ = (Ldown + lup_out) * f_up + Lside * f_cyl +
                  (((ln_lv + le_lv) + ls_lv) + lw_lv) * f_side;
    float Sstr = short_ * abs_k + long_ * abs_l;
    o_tmrt[cell] = sqrtf(sqrtf(Sstr / (abs_l * sbc))) - r2732;

    // ---- POI cardinals (lsideveg + dp) ----
    o_le[cell] = le_lv + ae;
    o_ls[cell] = ls_lv + aso;
    o_lw[cell] = lw_lv + aw;
    o_ln[cell] = ln_lv + an;
}

extern "C" __global__ void sw_rad_night_kernel(
    const unsigned char *__restrict__ sh_pb,
    const unsigned char *__restrict__ veg_pb,
    const unsigned char *__restrict__ vbsh_pb,
    const float *__restrict__ night_Lup,
    const float *__restrict__ ster, const float *__restrict__ psin,
    const float *__restrict__ pcos, const float *__restrict__ lsky_d2,
    const float *__restrict__ lsky_s2,
    const signed char *__restrict__ card_e, const signed char *__restrict__ card_s,
    const signed char *__restrict__ card_w, const signed char *__restrict__ card_n,
    const float *__restrict__ ccos_e, const float *__restrict__ ccos_s,
    const float *__restrict__ ccos_w, const float *__restrict__ ccos_n,
    double veg64, double shd64,
    float *__restrict__ o_tmrt, float *__restrict__ o_ldown,
    float *__restrict__ o_lside, float *__restrict__ o_le,
    float *__restrict__ o_ls, float *__restrict__ o_lw,
    float *__restrict__ o_ln,
    int rows, int cols, int n_patches) {

    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) {
        return;
    }
    const float one = 1.0f;
    const float half = 0.5f;
    const float f_cyl = 0.28f;
    const float f_up = 0.06f;
    const float f_side = 0.22f;
    const float abs_k = 0.7f;
    const float abs_l = 0.95f;
    const float sbc = 5.67051e-8f;
    const float r2732 = 273.2f;
    const float zero = 0.0f;
    int pstride = (n_patches + 7) / 8;

    float Lup = night_Lup[cell];
    float Ls_sky = 0.0f, Ld_sky = 0.0f;
    float Ls_veg = 0.0f, Ld_veg = 0.0f;
    float Ls_sh = 0.0f, Ld_sh = 0.0f;
    float ae = 0.0f, aso = 0.0f, aw = 0.0f, an = 0.0f;
    for (int idx = 0; idx < n_patches; ++idx) {
        bool shb = sw_pbit(sh_pb, cell, pstride, idx);
        bool vegb = sw_pbit(veg_pb, cell, pstride, idx);
        bool vbshb = sw_pbit(vbsh_pb, cell, pstride, idx);
        float sky_m = (shb && vegb) ? one : zero;
        double veg_m = ((!vegb) || (!vbshb)) ? 1.0 : 0.0;
        float sun_m = ((!shb) && vbshb) ? one : zero;
        Ld_sky = Ld_sky + sky_m * lsky_d2[idx];
        Ls_sky = Ls_sky + sky_m * lsky_s2[idx];
        double st64 = (double)ster[idx];
        double pc64 = (double)pcos[idx];
        double ps64 = (double)psin[idx];
        double Pv_c = (veg64 * st64) * pc64;
        double Pv_s = (veg64 * st64) * ps64;
        Ls_veg = (float)((double)Ls_veg + Pv_c * veg_m);
        Ld_veg = (float)((double)Ld_veg + Pv_s * veg_m);
        if (card_e[idx]) {
            ae = ae + (sky_m * lsky_s2[idx]) * ccos_e[idx];
            ae = (float)((double)ae + (Pv_c * (double)ccos_e[idx]) * veg_m);
        }
        if (card_s[idx]) {
            aso = aso + (sky_m * lsky_s2[idx]) * ccos_s[idx];
            aso = (float)((double)aso + (Pv_c * (double)ccos_s[idx]) * veg_m);
        }
        if (card_w[idx]) {
            aw = aw + (sky_m * lsky_s2[idx]) * ccos_w[idx];
            aw = (float)((double)aw + (Pv_c * (double)ccos_w[idx]) * veg_m);
        }
        if (card_n[idx]) {
            an = an + (sky_m * lsky_s2[idx]) * ccos_n[idx];
            an = (float)((double)an + (Pv_c * (double)ccos_n[idx]) * veg_m);
        }
        // night: solar_altitude <= 0 -> guard False everywhere -> else
        // branch (shaded term on temp_sh)
        double Psh_c = (shd64 * st64) * pc64;
        double Psh_s = (shd64 * st64) * ps64;
        double sm64 = (double)sun_m;
        Ls_sh = (float)((double)Ls_sh + Psh_c * sm64);
        Ld_sh = (float)((double)Ld_sh + Psh_s * sm64);
        if (card_e[idx]) {
            ae = (float)((double)ae + Psh_c * (double)ccos_e[idx] * sm64);
        }
        if (card_s[idx]) {
            aso = (float)((double)aso + Psh_c * (double)ccos_s[idx] * sm64);
        }
        if (card_w[idx]) {
            aw = (float)((double)aw + Psh_c * (double)ccos_w[idx] * sm64);
        }
        if (card_n[idx]) {
            an = (float)((double)an + Psh_c * (double)ccos_n[idx] * sm64);
        }
    }

    float ome = one - 0.9f;
    float pi32 = 3.14159274101257324f;  // fl32(pi)
    float refl = (((Ld_sky + Lup) * ome) * half) / pi32;
    float Ls_ref = 0.0f;
    float Ld_ref = 0.0f;
    for (int idx = 0; idx < n_patches; ++idx) {
        bool shb = sw_pbit(sh_pb, cell, pstride, idx);
        bool vegb = sw_pbit(veg_pb, cell, pstride, idx);
        bool vbshb = sw_pbit(vbsh_pb, cell, pstride, idx);
        float rm = ((!shb) || (!vegb) || (!vbshb)) ? one : zero;
        Ls_ref = Ls_ref + (((refl * ster[idx]) * pcos[idx]) * rm);
        Ld_ref = Ld_ref + (((refl * ster[idx]) * psin[idx]) * rm);
        if (card_e[idx]) {
            ae = ae + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_e[idx]);
        }
        if (card_s[idx]) {
            aso = aso + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_s[idx]);
        }
        if (card_w[idx]) {
            aw = aw + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_w[idx]);
        }
        if (card_n[idx]) {
            an = an + ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_n[idx]);
        }
    }

    float Lside = ((((Ls_sky + Ls_veg) + Ls_sh) + zero) + Ls_ref);
    float Ldown = ((((Ld_sky + Ld_veg) + Ld_sh) + zero) + Ld_ref);
    o_lside[cell] = Lside;
    o_ldown[cell] = Ldown;

    float le_lv = Lup * half;
    float ls_lv = Lup * half;
    float lw_lv = Lup * half;
    float ln_lv = Lup * half;
    float short_ = zero * f_cyl + (zero + zero) * f_up +
                   (((zero + zero) + zero) + zero) * f_side;
    float long_ = (Ldown + Lup) * f_up + Lside * f_cyl +
                  (((ln_lv + le_lv) + ls_lv) + lw_lv) * f_side;
    float Sstr = short_ * abs_k + long_ * abs_l;
    o_tmrt[cell] = sqrtf(sqrtf(Sstr / (abs_l * sbc))) - r2732;
    o_le[cell] = le_lv + ae;
    o_ls[cell] = ls_lv + aso;
    o_lw[cell] = lw_lv + aw;
    o_ln[cell] = ln_lv + an;
}
