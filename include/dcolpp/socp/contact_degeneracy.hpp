#pragma once
// dcolpp::socp -- contactDegeneracy: at a degenerate contact (shared
// face/edge/line, or a matched vertex) detect when the witness-point and/or
// contact-normal Jacobian is undefined. Two rank tests on the converged
// (s*, z*, G) -- no re-solve, no shape geometry. Also exports
// normalConeRank / BoundedActiveMat, reused by contact_manifold.hpp.

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <utility>

#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// Two rank-deficiency questions at the converged (x*, s*, z*):
//  - contact_manifold_dim = nx - rank(A), A = G^T (S^-1 Z) G
//    >0: multiple equally-optimal witness points; witness_point rows diverge
//  - normal_cone_dim = m - rank(A_active)
//    >0: contact normal isn't unique; normal_jacobian diverges
// Thresholds heuristic: > 0 means Jacobian rows unreliable.
struct ContactDegeneracy {
    int contact_manifold_dim = 0; // dim of the optimal witness-point set (0 = unique)
    int normal_cone_dim = 0;      // dim of the optimal dual/normal set (0 = unique)
    bool witness_jacobian_valid = true; // false iff contact_manifold_dim > 0
    bool normal_jacobian_valid = true;  // false iff normal_cone_dim > 0
};

// A_active's row count m is runtime (which constraints are active) but has a
// compile-time upper bound: <= 1 row per ORT constraint + 1 per active SOC
// block. A bounded dynamic-size Eigen matrix (Dynamic rows, MaxRows template
// arg) keeps that capacity on the stack, no heap. Shared with contactManifold.
template <int n_ort, int n_soc1, int n_soc2, int nx>
using BoundedActiveMat = Eigen::Matrix<double, Eigen::Dynamic, nx, 0,
                                        n_ort + (n_soc1 > 0 ? 1 : 0) + (n_soc2 > 0 ? 1 : 0), nx>;

// Builds A_active (see ContactDegeneracy's doc comment above) and its rank
// -> normal_cone_dim, without any heap allocation. Returns {m, rank}.
template <int n_ort, int n_soc1, int n_soc2, int nx>
std::pair<int, int> normalConeRank(const StackVec<n_ort, n_soc1, n_soc2>& z,
                                    const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G, double kActiveTol,
                                    double kRelZeroTol) {
    std::array<bool, n_ort> ort_active{};
    int m = 0;
    for (int i = 0; i < n_ort; ++i) {
        ort_active[i] = z(i) > kActiveTol;
        if (ort_active[i]) ++m;
    }
    bool soc1_active = false, soc2_active = false;
    if constexpr (n_soc1 > 0) {
        soc1_active = z.template segment<n_soc1>(n_ort).norm() > kActiveTol;
        if (soc1_active) ++m;
    }
    if constexpr (n_soc2 > 0) {
        soc2_active = z.template segment<n_soc2>(n_ort + n_soc1).norm() > kActiveTol;
        if (soc2_active) ++m;
    }

    BoundedActiveMat<n_ort, n_soc1, n_soc2, nx> A_active(m, nx);
    int mi = 0;
    for (int i = 0; i < n_ort; ++i) {
        if (ort_active[i]) A_active.row(mi++) = G.row(i);
    }
    if constexpr (n_soc1 > 0) {
        if (soc1_active) {
            A_active.row(mi++) =
                z.template segment<n_soc1>(n_ort).transpose() * G.template block<n_soc1, nx>(n_ort, 0);
        }
    }
    if constexpr (n_soc2 > 0) {
        if (soc2_active) {
            A_active.row(mi++) = z.template segment<n_soc2>(n_ort + n_soc1).transpose() *
                                  G.template block<n_soc2, nx>(n_ort + n_soc1, 0);
        }
    }

    // Rank only (the singular vectors are never used), so a rank-revealing
    // LU, not an SVD. setThreshold(kRelZeroTol) reproduces the prior
    // SVD criterion (sigma > kRelZeroTol * max(smax, 1)) on every tested
    // configuration.
    Eigen::FullPivLU<BoundedActiveMat<n_ort, n_soc1, n_soc2, nx>> lu(A_active);
    lu.setThreshold(kRelZeroTol);
    const int rAct = static_cast<int>(lu.rank());
    return {m, rAct};
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
ContactDegeneracy contactDegeneracy(const StackVec<n_ort, n_soc1, n_soc2>& s,
                                     const StackVec<n_ort, n_soc1, n_soc2>& z,
                                     const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G) {
    constexpr double kActiveTol = 1e-3;  // z_i / ||z_block|| above this -> "active"
    constexpr double kRelZeroTol = 1e-3; // A's own singular values, relative to G's scale

    // A = G^T (S^-1 Z) G -- same construction as computeSocpSensitivity;
    // fixed-size throughout, no MatrixXd.
    const PlainScaling<n_ort, n_soc1, n_soc2> Zs = plainScalingFromZ<n_ort, n_soc1, n_soc2>(z);
    const NTScaling<n_ort, n_soc1, n_soc2> Ss = scalingFromS<n_ort, n_soc1, n_soc2>(s);
    const PlainScaling<n_ort, n_soc1, n_soc2> SZ = solve(Ss, Zs);
    const Mat<n_ort + n_soc1 + n_soc2, nx> SZG = SZ.template applyMat<nx>(G);
    const Mat<nx, nx> A = G.transpose() * SZG;

    // Rank via FullPivLU (vectors unused here; contactManifold keeps SVD for
    // them). A's spectrum spans ~9 orders of magnitude and an active SOC
    // block's finite tangential eigenvalues must NOT be zeroed, so use an
    // absolute cutoff (kRelZeroTol * ||G||^2) converted into LU's relative
    // convention via maxPivot().
    const double g_scale = std::max(G.squaredNorm(), 1.0);
    Eigen::FullPivLU<Mat<nx, nx>> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = nx - static_cast<int>(luA.rank());

    const auto [m, rAct] = normalConeRank<n_ort, n_soc1, n_soc2, nx>(z, G, kActiveTol, kRelZeroTol);
    const int d_d = m - rAct;

    ContactDegeneracy out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = d_d;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (d_d == 0);
    return out;
}

} // namespace dcolpp::socp
