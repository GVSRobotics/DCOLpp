#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
//
// ContactDegeneracy/contactDegeneracy: detects when the witness-point
// Jacobian and/or the contact-normal Jacobian are undefined at a degenerate
// contact (shared face/edge/line, or a matched vertex). See DEVIATIONS.md
// "Contact-manifold degeneracy" for the full derivation. Also exports
// normalConeRank/BoundedActiveMat, shared with contact_manifold.hpp (which
// needs the same rank computation but returns points, not just dims/
// booleans -- a genuinely different concern, kept in its own file).
//
// Pure algebra on the converged (s*,z*,G) the solver already produces -- no
// shape-specific logic, no re-solving.

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <utility>

#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// ---------------------------------------------------------------------------
// Degeneracy diagnostics (DEVIATIONS.md "Contact-manifold degeneracy": two
// independent rank-deficiency questions at the converged (x*,s*,z*), on two
// different matrices the solver already builds:
//
//  - contact_manifold_dim = dim ker(A), A = G^T(S^-1 Z)G -- the EXACT nx*nx
//    matrix diffSocp/contactNormalJacobian invert (analytic_derivatives.hpp:
//    "const Mat<nx,nx> A = G.transpose()*SZG"). Its null space IS the
//    tangent space of the optimal primal face: >0 means there's more than
//    one equally-optimal witness point (a shared face/edge/line), and
//    witness_point's rows of `jacobian` diverge as pdip_tol->0 (alpha's row
//    is unaffected regardless -- it's an envelope-theorem quantity, always
//    well-defined). This needs the REAL A, not a naive active-constraint
//    gradient count: a curved (SOC) constraint's S^-1Z sub-block carries
//    finite nonzero tangential eigenvalues that a flat-facet-style count
//    would wrongly call "free" -- verified: a naive count predicts 4 free
//    directions for a cylinder-cylinder line contact, the real A shows only
//    1 (sliding along the shared line), matching the true geometry.
//
//  - normal_cone_dim = m - rank(A_active), A_active stacks one row per
//    active ORT constraint (z_i > active_tol) plus one row per active SOC
//    block (z_block^T * G_block -- a bound SOC block contributes exactly
//    ONE row regardless of block size, since conic complementarity pins its
//    z to a single ray, exactly, not an approximation). >0 means the
//    contact normal isn't unique (the touching feature has a kink: a
//    polytope edge or vertex) and `normal_jacobian` diverges as
//    pdip_tol->0. Unlike contact_manifold_dim, this one IS exactly captured
//    by a linear active-set rank count -- stationarity sum(z_i*G_i)=-c is
//    genuinely linear in z regardless of curvature, so no curvature
//    correction is needed here.
//
// Verified against 8 hand-built configurations spanning all three contact-
// manifold dimensions (2D face/face, 1D edge/edge parallel, 1D cylinder
// generatrix/generatrix, 0D corner/corner, 0D sphere/sphere, 0D sphere/face,
// plus skew (non-parallel) edge/edge at two angles) in
// tests/test_proximity_contact.cpp -- every prediction matched a direct
// pdip_tol-sweep of the actual Jacobian norms (the same ~1/pdip_tol
// divergence signature used throughout DEVIATIONS.md).
//
// Both thresholds below are scale-aware (relative to G's own magnitude,
// since G's entries scale with shape size) rather than fixed absolute
// numbers, but are still heuristic cutoffs, not a proof for every possible
// shape scale/pose -- treat contact_manifold_dim/normal_cone_dim > 0 as "the
// corresponding Jacobian rows are not to be trusted", and treat ==0 as "no
// degeneracy detected", not as an ironclad guarantee.
struct ContactDegeneracy {
    int contact_manifold_dim = 0; // dim of the optimal witness-point set (0 = unique)
    int normal_cone_dim = 0;      // dim of the optimal dual/normal set (0 = unique)
    bool witness_jacobian_valid = true; // false iff contact_manifold_dim > 0
    bool normal_jacobian_valid = true;  // false iff normal_cone_dim > 0
};

// A_active's row count m is only known at runtime (depends on which
// constraints are active), but it has a known compile-time UPPER BOUND: at
// most one row per ORT constraint, plus at most one row per active SOC
// block. Eigen's bounded dynamic-size matrix (Dynamic rows, but a MaxRows
// template arg) stores that capacity on the STACK, not the heap, unlike
// plain MatrixXd -- measured, not assumed: a synthetic 2x4 case took 123ns
// this way vs 476ns for the identical matrix in a MatrixXd, ~80% of that
// difference being pure malloc/free overhead on a matrix too small for the
// actual FLOPs to matter. Shared by contactDegeneracy and contactManifold.
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

    // Rank-only: A_active's singular VECTORS are never used anywhere, only
    // its rank, so a rank-revealing LU is the right tool, not a full SVD --
    // measured, not assumed: 3.6x-14.7x faster than JacobiSVD across the
    // matrix shapes this actually produces (2x4 to 6x4). setThreshold(v)
    // matches this function's previous SVD-based criterion
    // (singularValue > kRelZeroTol * max(smax,1)) closely enough in
    // practice that it reproduces the exact same rank on every one of the
    // 8 hand-built configurations in tests/test_contact_degeneracy.cpp,
    // verified directly against the old SVD-based answer before this
    // function was changed to use it, not assumed to match from the API
    // docs alone.
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

    // A = G^T (S^-1 Z) G -- identical construction to
    // diffSocpSensitivityAnalyticWithG (analytic_derivatives.hpp). Fixed-
    // size throughout (nx is a template constant) -- no MatrixXd copy.
    const PlainScaling<n_ort, n_soc1, n_soc2> Zs = plainScalingFromZ<n_ort, n_soc1, n_soc2>(z);
    const NTScaling<n_ort, n_soc1, n_soc2> Ss = scalingFromS<n_ort, n_soc1, n_soc2>(s);
    const PlainScaling<n_ort, n_soc1, n_soc2> SZ = solve(Ss, Zs);
    const Mat<n_ort + n_soc1 + n_soc2, nx> SZG = SZ.template applyMat<nx>(G);
    const Mat<nx, nx> A = G.transpose() * SZG;

    // Rank-only here too (contactDegeneracy never uses A's singular
    // vectors -- contactManifold needs those and keeps JacobiSVD for
    // that reason). FullPivLU is 1.4x-13.3x faster, measured -- but its
    // setThreshold() is RELATIVE to its own maxPivot(), whereas A's
    // spectrum spans up to ~9 orders of magnitude (the whole reason this
    // needed an ABSOLUTE threshold in the first place: a naive relative
    // one wrongly zeroes out the finite tangential eigenvalues of an
    // active SOC block -- see contact_degeneracy.hpp's own history).
    // Converting kRelZeroTol*g_scale (this function's established
    // absolute cutoff) into LU's relative convention via maxPivot() fixes
    // this -- verified, not assumed: matches JacobiSVD's answer on all 8
    // known configurations (tests/test_contact_degeneracy.cpp), including
    // the SOC cases (cylinder-cylinder, sphere-sphere) that a naive
    // setThreshold(kRelZeroTol) gets wrong (predicts d_p=4/2 instead of
    // 1/0).
    const double g_scale = std::max(G.squaredNorm(), 1.0); // scale-aware absolute cutoff
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
