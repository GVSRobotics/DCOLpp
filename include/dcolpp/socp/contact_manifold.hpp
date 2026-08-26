#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
//
// ContactManifold/contactManifold: when contact_manifold_dim > 0 (a shared
// face/edge/line, not a single point), a single witness point
// under-represents the true contact region -- multiple points are the
// standard input a downstream contact resolver (NCP, LCP, etc.) needs for a
// stable resolved contact (this is exactly why physics engines like
// Bullet/PhysX build multi-point "contact manifolds" instead of trusting
// one point for face contacts). Reuses contact_degeneracy.hpp's
// normalConeRank for normal_cone_dim, but is a genuinely different concern
// from ContactDegeneracy (points, not just dims/booleans) and lives in its
// own file for that reason.
//
// dim 0: the single witness point, unchanged.
// dim 1: the segment's two endpoints -- exact and, importantly, provably
//   independent of where x* happens to sit on the degenerate line: clipping
//   forward/backward from ANY point on a line against every constraint
//   recovers the same absolute endpoints (shifting the start point by s
//   along the line shifts both t_min and t_max by -s, so x*+t_min*d is
//   unchanged). No centering needed for this case.
// dim 2: NOT the case above -- x* CAN sit off-center within a 2D degenerate
//   patch (verified: SocpInitStrategy::Geometric's shape-seeded initial
//   guess measurably biases x* off the true analytic center of an
//   asymmetric overlap, while Generic lands on it exactly -- so trusting
//   x* blindly here would inherit that bias into a naive point sweep).
//   Two stages, both built from the SAME clip primitive as dim 1:
//     1. Recenter: clip +-d1 through x* -> exact center of that 1D slice,
//        c1; then clip +-d2 through c1 -> c2. Two applications of the exact
//        dim-1 trick, not a general LP -- cheap, and a real (if imperfect
//        for very eccentric patches -- not exact in 2D the way it is in
//        1D) improvement over trusting x* raw.
//     2. Oversample M=max(2K,8) angularly-spaced rays from c2, then reduce
//        to K via greedy farthest-point selection (maximize the minimum
//        distance to already-picked points) -- avoids clustering the
//        output on one side of an elongated patch, which fixed small-K
//        angular sampling from an off-center point would otherwise do.
//
// K (default 4, the common "quad manifold" convention) is only used for
// dim==2; dim 0/1 always return 1/2 points regardless. K can be increased
// for finer resolution -- cost scales as O(K) (oversample+reduce), still
// closed-form ray clips throughout, no LP solver.
//
// Superset of ContactDegeneracy (also returns contact_manifold_dim/
// normal_cone_dim/the two *_valid booleans) so a caller doesn't need to
// call both -- but its own SVD-of-A requests the V matrix (needed for the
// null-space basis), which contactDegeneracy's does not, so calling both
// when only the manifold is wanted duplicates work; use contactDegeneracy
// alone if the manifold points themselves aren't needed.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "dcolpp/socp/contact_degeneracy.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

namespace detail {
constexpr double kTwoPi = 6.28318530717958647692; // avoid relying on M_PI (not defined by MSVC/some mingw setups)
} // namespace detail

struct ContactManifold {
    std::vector<Eigen::Vector3d> witness_points; // reference (shape-1) frame; size 1, 2, or K
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
};

template <int n_ort, int n_soc1, int n_soc2, int nx>
ContactManifold contactManifold(const DecisionVec<nx>& x, const StackVec<n_ort, n_soc1, n_soc2>& s,
                                 const StackVec<n_ort, n_soc1, n_soc2>& z,
                                 const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G, int K = 4) {
    using Eigen::VectorXd;
    using SVec = StackVec<n_ort, n_soc1, n_soc2>;
    using XVec = Vec<nx>;
    constexpr double kRelZeroTol = 1e-3;
    constexpr double kActiveTol = 1e-3;

    // A = G^T(S^-1 Z)G. Fixed-size throughout -- no MatrixXd copy.
    const PlainScaling<n_ort, n_soc1, n_soc2> Zs = plainScalingFromZ<n_ort, n_soc1, n_soc2>(z);
    const NTScaling<n_ort, n_soc1, n_soc2> Ss = scalingFromS<n_ort, n_soc1, n_soc2>(s);
    const PlainScaling<n_ort, n_soc1, n_soc2> SZ = solve(Ss, Zs);
    const Mat<n_ort + n_soc1 + n_soc2, nx> SZG = SZ.template applyMat<nx>(G);
    const Mat<nx, nx> A = G.transpose() * SZG;
    const double g_scale = std::max(G.squaredNorm(), 1.0);

    // Rank first, cheaply, via FullPivLU (same construction as
    // contactDegeneracy -- see its comment for why the threshold needs the
    // maxPivot() conversion). dim==0 is the common case in practice
    // (non-degenerate contacts, not exactly touching face-to-face/edge-to-
    // edge) and never needs A's singular VECTORS at all -- computing them
    // unconditionally here was a real, measured inefficiency (e.g. box
    // corner-corner: ~660ns via JacobiSVD-with-V vs ~45ns via this rank
    // check alone, a case where d_p=0 so those vectors were always wasted
    // work), not just a theoretical one; only construct the (materially
    // more expensive) JacobiSVD-with-V below once d_p>0 confirms they're
    // actually needed.
    Eigen::FullPivLU<Mat<nx, nx>> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = nx - static_cast<int>(luA.rank());

    // normal_cone_dim: shared with contactDegeneracy (contact_degeneracy.hpp).
    const auto [m, rAct] = normalConeRank<n_ort, n_soc1, n_soc2, nx>(z, G, kActiveTol, kRelZeroTol);
    const int d_d = m - rAct;

    ContactManifold out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = d_d;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (d_d == 0);

    if (d_p == 0) {
        out.witness_points.push_back(x.template head<3>());
        return out;
    }

    // d_p > 0: NOW pay for the null-space basis -- the one thing FullPivLU
    // doesn't give without an extra (unvalidated here) orthonormalization
    // step, so JacobiSVD stays the tool for this part specifically.
    Eigen::JacobiSVD<Mat<nx, nx>> svdA(A, Eigen::ComputeFullV);

    // Closed-form ray clip: walk point + t*dir, find the largest t-interval
    // for which every ORT row and every SOC block stays feasible. s_here is
    // the slack (h - G*point) AT `point` -- since s = h - Gx and moving by
    // t*dir changes s linearly (s(t) = s_here - t*(G*dir)), no need to
    // carry h separately.
    auto clip = [&](const SVec& s_here, const XVec& dir) -> std::pair<double, double> {
        double t_min = -1e18, t_max = 1e18;
        for (int i = 0; i < n_ort; ++i) {
            const double Gid = G.row(i).dot(dir);
            if (std::abs(Gid) < 1e-14) continue;
            const double t_bound = s_here(i) / Gid;
            if (Gid > 0) t_max = std::min(t_max, t_bound);
            else t_min = std::max(t_min, t_bound);
        }
        auto clipSoc = [&](int off, int dim) {
            if (dim == 0) return;
            const VectorXd s0v = s_here.segment(off, dim);
            const VectorXd a = G.middleRows(off, dim) * dir;
            // If `dir` doesn't move this SOC block's constraint at all (a
            // is negligible relative to s0v's own scale), there's nothing
            // to clip against -- skip entirely, BEFORE computing A_c/B_c/
            // C_c. This matters most exactly where dir is tangent to an
            // active (touching) SOC block's own smooth boundary (e.g. two
            // parallel cylinders/capsules sliding along their shared
            // touching line: that line is tangent to each shape's own
            // radial constraint, so a is mathematically zero there) --
            // measured, not assumed: for cylinder-cylinder/capsule-capsule/
            // cylinder-capsule with an axial offset, `a` lands at ~1e-8
            // (roundoff amplified through the near-zero-singular-value
            // null direction), not true machine epsilon. A_c/B_c/C_c have
            // DIFFERENT natural magnitudes (O(a^2), O(s0*a), O(s0^2)), so
            // a single fixed absolute threshold on the derived B_c isn't a
            // reliable zero test -- it previously let a ~1e-13 B_c (itself
            // a product of two already-tiny quantities) through as
            // "significant", dividing C_c by it and producing witness
            // points off by 5+ orders of magnitude. Checking `a` itself,
            // relative to a real scale, catches this before it can happen.
            if (a.norm() < 1e-6 * std::max(s0v.norm(), 1.0)) return;
            const double s0 = s0v(0), a0 = a(0);
            const VectorXd s_tail = s0v.tail(dim - 1);
            const VectorXd a_tail = a.tail(dim - 1);
            const double A_c = a0 * a0 - a_tail.squaredNorm();
            const double B_c = -2.0 * (s0 * a0 - s_tail.dot(a_tail));
            const double C_c = s0 * s0 - s_tail.squaredNorm();
            if (std::abs(A_c) < 1e-14) {
                if (std::abs(B_c) > 1e-14) {
                    const double t_root = -C_c / B_c;
                    if (B_c > 0) t_max = std::min(t_max, t_root);
                    else t_min = std::max(t_min, t_root);
                }
                return;
            }
            const double disc = B_c * B_c - 4 * A_c * C_c;
            if (disc < 0) return;
            const double sq = std::sqrt(disc);
            double r1 = (-B_c - sq) / (2 * A_c), r2 = (-B_c + sq) / (2 * A_c);
            if (r1 > r2) std::swap(r1, r2);
            if (A_c < 0) { // feasible BETWEEN roots (typical: bounded cone cross-section)
                t_min = std::max(t_min, r1);
                t_max = std::min(t_max, r2);
            } // A_c>0 (feasible outside [r1,r2]) not expected for a bounded shape; left unhandled
        };
        clipSoc(n_ort, n_soc1);
        clipSoc(n_ort + n_soc1, n_soc2);
        return {t_min, t_max};
    };
    auto shiftSlack = [&](const SVec& s_here, const XVec& dir, double t) -> SVec {
        return SVec(s_here - t * (G * dir));
    };

    const XVec d1 = svdA.matrixV().col(nx - 1); // smallest singular value's direction
    if (d_p == 1) {
        const auto [t_min, t_max] = clip(s, d1);
        out.witness_points.push_back(XVec(x + t_min * d1).template head<3>());
        out.witness_points.push_back(XVec(x + t_max * d1).template head<3>());
        return out;
    }

    // d_p >= 2: use the two smallest directions to span the degenerate
    // plane (a d_p>2 case -- a 3D+ degenerate set -- would need a genuinely
    // different treatment; not expected for any of the 7 shapes here, so
    // not handled beyond this 2D plane).
    const XVec d2 = svdA.matrixV().col(nx - 2);

    // Stage 1: recenter via two sequential exact 1D centerings.
    const auto [t1min, t1max] = clip(s, d1);
    const double t1c = 0.5 * (t1min + t1max);
    const XVec c1 = x + t1c * d1;
    const SVec s1 = shiftSlack(s, d1, t1c);

    const auto [t2min, t2max] = clip(s1, d2);
    const double t2c = 0.5 * (t2min + t2max);
    const XVec c2 = c1 + t2c * d2;
    const SVec s2 = shiftSlack(s1, d2, t2c);

    // Stage 2: oversample, then greedily reduce to K by farthest-point
    // selection (maximize the minimum distance to already-picked points),
    // so the output spans the patch instead of clustering.
    K = std::max(K, 3);
    const int M = std::max(2 * K, 8);
    std::vector<Eigen::Vector3d> candidates;
    candidates.reserve(M);
    for (int mi = 0; mi < M; ++mi) {
        const double theta = detail::kTwoPi * mi / M;
        const XVec dir = std::cos(theta) * d1 + std::sin(theta) * d2;
        const auto [tm_min, tm_max] = clip(s2, dir);
        (void)tm_min; // sampling outward from an interior point: only the forward hit matters
        candidates.push_back(XVec(c2 + tm_max * dir).template head<3>());
    }

    std::vector<bool> used(candidates.size(), false);
    // seed: farthest candidate from the center.
    int best = 0;
    double best_d = -1.0;
    const Eigen::Vector3d center3 = c2.template head<3>();
    for (int mi = 0; mi < static_cast<int>(candidates.size()); ++mi) {
        const double d = (candidates[mi] - center3).squaredNorm();
        if (d > best_d) { best_d = d; best = mi; }
    }
    used[best] = true;
    out.witness_points.push_back(candidates[best]);
    while (static_cast<int>(out.witness_points.size()) < K) {
        int next = -1;
        double next_d = -1.0;
        for (int mi = 0; mi < static_cast<int>(candidates.size()); ++mi) {
            if (used[mi]) continue;
            double min_d = 1e300;
            for (const auto& p : out.witness_points) min_d = std::min(min_d, (candidates[mi] - p).squaredNorm());
            if (min_d > next_d) { next_d = min_d; next = mi; }
        }
        if (next < 0) break; // fewer candidates than requested K (M<K, shouldn't happen given M>=2K)
        used[next] = true;
        out.witness_points.push_back(candidates[next]);
    }
    return out;
}

} // namespace dcolpp::socp
