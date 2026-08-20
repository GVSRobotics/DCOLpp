#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/solvers/coneqp/NT_scaling_chol_2.jl (Kevin Tracy, MIT License).
// See NOTICE.md at the repository root for full attribution.
//
// Nesterov-Todd (NT) scaling: the block-diagonal operator W (ORT block:
// elementwise sqrt(s/z); SOC blocks: the NT scaling matrix, stored
// Cholesky-factored) used throughout the interior-point solver to rescale
// the KKT system each iteration. `PlainScaling` is the un-factored
// block-diagonal cone-multiplier operator (Z = blockdiag(diag(z_ort),
// arrow(z_soc1), arrow(z_soc2))) needed later for the implicit-function-
// theorem sensitivity (`dcolpp::socp::diffSocp`).

#include <Eigen/Dense>
#include "dcolpp/socp/cone_utils.hpp"

namespace dcolpp::socp {

template <int n_ort, int n_soc1, int n_soc2>
struct NTScaling {
    Vec<n_ort> ort;
    Mat<n_soc1, n_soc1> soc1;
    Eigen::LLT<Mat<n_soc1, n_soc1>> soc1_fact;
    Mat<n_soc2, n_soc2> soc2;
    Eigen::LLT<Mat<n_soc2, n_soc2>> soc2_fact;

    // W \ g
    StackVec<n_ort, n_soc1, n_soc2> solve(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) out.template head<n_ort>() = g.template head<n_ort>().cwiseQuotient(ort);
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1_fact.solve(g.template segment<n_soc1>(n_ort));
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2_fact.solve(g.template segment<n_soc2>(n_ort + n_soc1));
        return out;
    }

    // W * g
    StackVec<n_ort, n_soc1, n_soc2> apply(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) out.template head<n_ort>() = g.template head<n_ort>().cwiseProduct(ort);
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1 * g.template segment<n_soc1>(n_ort);
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2 * g.template segment<n_soc2>(n_ort + n_soc1);
        return out;
    }

    // W \ G  (columnwise)
    template <int NX>
    Mat<n_ort + n_soc1 + n_soc2, NX> solveMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
        Mat<n_ort + n_soc1 + n_soc2, NX> out;
        for (int j = 0; j < NX; ++j) out.col(j) = solve(G.col(j));
        return out;
    }

    // W * G  (columnwise)
    template <int NX>
    Mat<n_ort + n_soc1 + n_soc2, NX> applyMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
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

    StackVec<n_ort, n_soc1, n_soc2> apply(const StackVec<n_ort, n_soc1, n_soc2>& g) const {
        StackVec<n_ort, n_soc1, n_soc2> out;
        if constexpr (n_ort > 0) out.template head<n_ort>() = g.template head<n_ort>().cwiseProduct(ort);
        if constexpr (n_soc1 > 0) out.template segment<n_soc1>(n_ort) = soc1 * g.template segment<n_soc1>(n_ort);
        if constexpr (n_soc2 > 0) out.template segment<n_soc2>(n_ort + n_soc1) = soc2 * g.template segment<n_soc2>(n_ort + n_soc1);
        return out;
    }

    template <int NX>
    Mat<n_ort + n_soc1 + n_soc2, NX> applyMat(const Mat<n_ort + n_soc1 + n_soc2, NX>& G) const {
        Mat<n_ort + n_soc1 + n_soc2, NX> out;
        for (int j = 0; j < NX; ++j) out.col(j) = apply(G.col(j));
        return out;
    }
};

// W \ Z  (NTScaling \ PlainScaling -> PlainScaling), used by the implicit
// differentiation solve in Phase 3.
template <int n_ort, int n_soc1, int n_soc2>
PlainScaling<n_ort, n_soc1, n_soc2> solve(const NTScaling<n_ort, n_soc1, n_soc2>& W,
                                           const PlainScaling<n_ort, n_soc1, n_soc2>& Z) {
    PlainScaling<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) out.ort = Z.ort.cwiseQuotient(W.ort);
    if constexpr (n_soc1 > 0) out.soc1 = W.soc1_fact.solve(Z.soc1);
    if constexpr (n_soc2 > 0) out.soc2 = W.soc2_fact.solve(Z.soc2);
    return out;
}

template <int n_soc>
Mat<n_soc, n_soc> socNTScaling(const Vec<n_soc>& s_soc, const Vec<n_soc>& z_soc) {
    if constexpr (n_soc == 0) {
        return Mat<0, 0>{};
    } else {
        const Vec<n_soc> zbar = normalize_soc<n_soc>(z_soc);
        const Vec<n_soc> sbar = normalize_soc<n_soc>(s_soc);
        const double gamma = std::sqrt((1.0 + zbar.dot(sbar)) / 2.0);

        Vec<n_soc> zbar_flipped = -zbar;
        zbar_flipped(0) = zbar(0);
        const Vec<n_soc> wbar = (sbar + zbar_flipped) / (2.0 * gamma);

        const double b = 1.0 / (wbar(0) + 1.0);
        Mat<n_soc, n_soc> Wbar = Mat<n_soc, n_soc>::Zero();
        Wbar.row(0) = wbar.transpose();
        Wbar.template block<n_soc - 1, 1>(1, 0) = wbar.template tail<n_soc - 1>();
        Wbar.template block<n_soc - 1, n_soc - 1>(1, 1) =
            Mat<n_soc - 1, n_soc - 1>::Identity() +
            b * (wbar.template tail<n_soc - 1>() * wbar.template tail<n_soc - 1>().transpose());

        const double eta = std::pow(soc_quad_J<n_soc>(s_soc) / soc_quad_J<n_soc>(z_soc), 0.25);
        return eta * Wbar;
    }
}

// Z = blockdiag(diag(z_ort), arrow(z_soc1), arrow(z_soc2)) -- the plain
// (non-NT) cone-multiplier operator used by the implicit-function-theorem
// sensitivity (dcolpp::socp::diffSocp, Phase 3).
template <int n_ort, int n_soc1, int n_soc2>
PlainScaling<n_ort, n_soc1, n_soc2> plainScalingFromZ(const StackVec<n_ort, n_soc1, n_soc2>& z) {
    PlainScaling<n_ort, n_soc1, n_soc2> Z;
    if constexpr (n_ort > 0) Z.ort = z.template head<n_ort>();
    if constexpr (n_soc1 > 0) Z.soc1 = arrow<n_soc1, double>(z.template segment<n_soc1>(n_ort));
    if constexpr (n_soc2 > 0) Z.soc2 = arrow<n_soc2, double>(z.template segment<n_soc2>(n_ort + n_soc1));
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
NTScaling<n_ort, n_soc1, n_soc2> scalingFromS(const StackVec<n_ort, n_soc1, n_soc2>& s) {
    NTScaling<n_ort, n_soc1, n_soc2> S;
    if constexpr (n_ort > 0) S.ort = s.template head<n_ort>();
    if constexpr (n_soc1 > 0) {
        S.soc1 = arrow<n_soc1, double>(s.template segment<n_soc1>(n_ort));
        S.soc1_fact = Eigen::LLT<Mat<n_soc1, n_soc1>>(S.soc1.template selfadjointView<Eigen::Upper>());
    }
    if constexpr (n_soc2 > 0) {
        S.soc2 = arrow<n_soc2, double>(s.template segment<n_soc2>(n_ort + n_soc1));
        S.soc2_fact = Eigen::LLT<Mat<n_soc2, n_soc2>>(S.soc2.template selfadjointView<Eigen::Upper>());
    }
    return S;
}

template <int n_ort, int n_soc1, int n_soc2>
NTScaling<n_ort, n_soc1, n_soc2> calcNTScalings(const StackVec<n_ort, n_soc1, n_soc2>& s,
                                                 const StackVec<n_ort, n_soc1, n_soc2>& z) {
    NTScaling<n_ort, n_soc1, n_soc2> W;
    if constexpr (n_ort > 0) {
        W.ort = (s.template head<n_ort>().cwiseQuotient(z.template head<n_ort>())).cwiseSqrt();
    }
    if constexpr (n_soc1 > 0) {
        W.soc1 = socNTScaling<n_soc1>(s.template segment<n_soc1>(n_ort), z.template segment<n_soc1>(n_ort));
        W.soc1_fact = Eigen::LLT<Mat<n_soc1, n_soc1>>(W.soc1.template selfadjointView<Eigen::Upper>());
    }
    if constexpr (n_soc2 > 0) {
        W.soc2 = socNTScaling<n_soc2>(s.template segment<n_soc2>(n_ort + n_soc1), z.template segment<n_soc2>(n_ort + n_soc1));
        W.soc2_fact = Eigen::LLT<Mat<n_soc2, n_soc2>>(W.soc2.template selfadjointView<Eigen::Upper>());
    }
    return W;
}

} // namespace dcolpp::socp
