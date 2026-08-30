#pragma once
// dcolpp::socp
//
// Nesterov-Todd (NT) scaling: block-diagonal operator W rescaling the KKT
// system (ORT: elementwise sqrt(s/z); SOC: NT scaling matrix, Cholesky-factored).
// PlainScaling is the un-factored cone-multiplier Z for implicit-function-theorem
// sensitivity. SOC blocks use Cholesky factors (never explicit inverses) to avoid
// rho ~ 0 roundoff amplification near convergence. Batch multiple RHS through
// one factorization instead (solveMat).

#include <Eigen/Dense>
#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/small_llt.hpp"

namespace dcolpp::socp {

template <int n_ort, int n_soc1, int n_soc2>
struct NTScaling {
    Vec<n_ort> ort;
    Mat<n_soc1, n_soc1> soc1;
    SmallLLT<n_soc1> soc1_fact;
    Mat<n_soc2, n_soc2> soc2;
    SmallLLT<n_soc2> soc2_fact;

    // W \ g
    DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> solve(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) {
            for (int i = 0; i < n_ort; ++i) out(i) = g(i) / ort(i);
        }
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1_fact.solve(Vec<n_soc1>(g.template segment<n_soc1>(n_ort)));
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2_fact.solve(Vec<n_soc2>(g.template segment<n_soc2>(n_ort + n_soc1)));
        return out;
    }

    // W * g
    DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> apply(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) {
            for (int i = 0; i < n_ort; ++i) out(i) = g(i) * ort(i);
        }
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1 * g.template segment<n_soc1>(n_ort);
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2 * g.template segment<n_soc2>(n_ort + n_soc1);
        return out;
    }

    // W \ G, all columns through one Cholesky factorization each (not a
    // per-column solve() loop -- same numerics, fewer factor-solve calls).
    template <int NX>
    DCOLPP_INLINE Mat<n_ort + n_soc1 + n_soc2, NX> solveMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
        Mat<n_ort + n_soc1 + n_soc2, NX> out;
        if constexpr (n_ort > 0) {
            for (int i = 0; i < n_ort; ++i) {
                const double inv_ort_i = 1.0 / ort(i);
                for (int j = 0; j < NX; ++j) out(i, j) = inv_ort_i * G(i, j);
            }
        }
        if constexpr (n_soc1 > 0) out.template middleRows<n_soc1>(n_ort) = soc1_fact.solve(G.template middleRows<n_soc1>(n_ort));
        if constexpr (n_soc2 > 0) out.template middleRows<n_soc2>(n_ort + n_soc1) = soc2_fact.solve(G.template middleRows<n_soc2>(n_ort + n_soc1));
        return out;
    }

    // W * G  (columnwise)
    template <int NX>
    DCOLPP_INLINE Mat<n_ort + n_soc1 + n_soc2, NX> applyMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
        Mat<n_ort + n_soc1 + n_soc2, NX> out;
        for (int j = 0; j < NX; ++j) out.col(j) = apply(G.col(j));
        return out;
    }
};

// Un-factored block-diagonal cone operator: blockdiag(diag(ort), soc1, soc2).
template <int n_ort, int n_soc1, int n_soc2>
struct PlainScaling {
    Vec<n_ort> ort;
    Mat<n_soc1, n_soc1> soc1;
    Mat<n_soc2, n_soc2> soc2;

    DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> apply(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) {
            for (int i = 0; i < n_ort; ++i) out(i) = g(i) * ort(i);
        }
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1 * g.template segment<n_soc1>(n_ort);
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2 * g.template segment<n_soc2>(n_ort + n_soc1);
        return out;
    }

    template <int NX>
    DCOLPP_INLINE Mat<n_ort + n_soc1 + n_soc2, NX> applyMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
        Mat<n_ort + n_soc1 + n_soc2, NX> out;
        for (int j = 0; j < NX; ++j) out.col(j) = apply(G.col(j));
        return out;
    }
};

// W \ Z  (NTScaling \ PlainScaling -> PlainScaling), used by the implicit
// differentiation solve in Phase 3.
template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE PlainScaling<n_ort, n_soc1, n_soc2> solve(const NTScaling<n_ort, n_soc1, n_soc2>& W,
                                           const PlainScaling<n_ort, n_soc1, n_soc2>& Z) {
    PlainScaling<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) out.ort = Z.ort.cwiseQuotient(W.ort);
    if constexpr (n_soc1 > 0) out.soc1 = W.soc1_fact.solve(Z.soc1);
    if constexpr (n_soc2 > 0) out.soc2 = W.soc2_fact.solve(Z.soc2);
    return out;
}

template <int n_soc>
DCOLPP_INLINE Mat<n_soc, n_soc> socNTScaling(const Vec<n_soc>& s_soc, const Vec<n_soc>& z_soc) {
    if constexpr (n_soc == 0) {
        return Mat<0, 0>{};
    } else {
        const Vec<n_soc> zbar = normalize_soc<n_soc>(z_soc);
        const Vec<n_soc> sbar = normalize_soc<n_soc>(s_soc);
        double zs_dot = 0.0;
        for (int i = 0; i < n_soc; ++i) zs_dot += zbar(i) * sbar(i);
        const double gamma = std::sqrt((1.0 + zs_dot) / 2.0);
        const double inv_2gamma = 1.0 / (2.0 * gamma);

        // wbar = (sbar + [zbar(0); -zbar(1:)]) / (2*gamma)
        Vec<n_soc> wbar;
        wbar(0) = (sbar(0) + zbar(0)) * inv_2gamma;
        for (int i = 1; i < n_soc; ++i) wbar(i) = (sbar(i) - zbar(i)) * inv_2gamma;

        const double b = 1.0 / (wbar(0) + 1.0);
        const double eta = std::sqrt(std::sqrt(soc_quad_J<n_soc>(s_soc) / soc_quad_J<n_soc>(z_soc)));

        // Wbar = [wbar'; wbar(1:) (I + b*wbar(1:)*wbar(1:)')], then eta*Wbar.
        Mat<n_soc, n_soc> W;
        W(0, 0) = eta * wbar(0);
        for (int i = 1; i < n_soc; ++i) {
            W(0, i) = eta * wbar(i);
            W(i, 0) = eta * wbar(i);
        }
        for (int i = 1; i < n_soc; ++i) {
            for (int j = 1; j < n_soc; ++j) {
                W(i, j) = eta * ((i == j ? 1.0 : 0.0) + b * wbar(i) * wbar(j));
            }
        }
        return W;
    }
}

// Z = blockdiag(diag(z_ort), arrow(z_soc1), arrow(z_soc2)) -- the plain
// (non-NT) cone-multiplier operator used by the implicit-function-theorem
// sensitivity (dcolpp::socp::diffSocp, Phase 3).
template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE PlainScaling<n_ort, n_soc1, n_soc2> plainScalingFromZ(const StackVec<n_ort, n_soc1, n_soc2>& z) {
    PlainScaling<n_ort, n_soc1, n_soc2> Z;
    if constexpr (n_ort > 0) Z.ort = z.template head<n_ort>();
    if constexpr (n_soc1 > 0) Z.soc1 = arrow<n_soc1>(z.template segment<n_soc1>(n_ort));
    if constexpr (n_soc2 > 0) Z.soc2 = arrow<n_soc2>(z.template segment<n_soc2>(n_ort + n_soc1));
    return Z;
}

// S = blockdiag(diag(s_ort), arrow(s_soc1), arrow(s_soc2)), Cholesky-factored
// -- NOT the Nesterov-Todd scaling (that's calcNTScalings below, built from
// BOTH s and z via the geometric-mean formula). This reuses the NTScaling
// struct shape for a different purpose, exactly as
// src/proximity.jl's `diff_socp` does: `S = NT_scaling_2(s[idx_ort],
// arrow(s[idx_soc1]), cholesky(arrow(s[idx_soc1])), ...)` builds it directly
// from `s` alone, never calling calc_NT_scalings. Used by
// dcolpp::socp::diffSocp (Phase 3); mixing this up with the true NT scaling
// silently gives a plausible-looking but wrong sensitivity.
template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE NTScaling<n_ort, n_soc1, n_soc2> scalingFromS(const StackVec<n_ort, n_soc1, n_soc2>& s) {
    NTScaling<n_ort, n_soc1, n_soc2> S;
    if constexpr (n_ort > 0) S.ort = s.template head<n_ort>();
    if constexpr (n_soc1 > 0) {
        S.soc1 = arrow<n_soc1>(s.template segment<n_soc1>(n_ort));
        S.soc1_fact.compute(S.soc1);
    }
    if constexpr (n_soc2 > 0) {
        S.soc2 = arrow<n_soc2>(s.template segment<n_soc2>(n_ort + n_soc1));
        S.soc2_fact.compute(S.soc2);
    }
    return S;
}

template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE NTScaling<n_ort, n_soc1, n_soc2> calcNTScalings(const StackVec<n_ort, n_soc1, n_soc2>& s,
                                                 const StackVec<n_ort, n_soc1, n_soc2>& z) {
    NTScaling<n_ort, n_soc1, n_soc2> W;
    if constexpr (n_ort > 0) {
        W.ort = (s.template head<n_ort>().cwiseQuotient(z.template head<n_ort>())).cwiseSqrt();
    }
    if constexpr (n_soc1 > 0) {
        W.soc1 = socNTScaling<n_soc1>(s.template segment<n_soc1>(n_ort), z.template segment<n_soc1>(n_ort));
        W.soc1_fact.compute(W.soc1);
    }
    if constexpr (n_soc2 > 0) {
        W.soc2 = socNTScaling<n_soc2>(s.template segment<n_soc2>(n_ort + n_soc1), z.template segment<n_soc2>(n_ort + n_soc1));
        W.soc2_fact.compute(W.soc2);
    }
    return W;
}

} // namespace dcolpp::socp
