#pragma once
// dcolpp::socp -- warm-starting for temporally-continuous proximity queries.
//
// A physics step, a trajectory-optimization inner loop, or a smooth ergodic
// sweep calls proximityJacobian repeatedly at poses `g` that differ only
// slightly between calls (identically so for a body at rest or a
// fixed-base grasp). Re-solving the SOCP from the cold geometric guess
// every time rediscovers the whole solution -- including the active set,
// which for sustained contact never changes. This header lets the caller
// carry the previous converged (x, s, z, mu) forward in a small per-pair
// handle; proximityJacobian(..., ContactWarmState*) then seeds the solve
// from it (warmStartInit, socp_init.hpp): 1 iteration when the pose
// did not move (rest / grasp -> 3-6x faster), ~3-5 when it moved a little
// (~1.5-2x), and it auto-falls back to the cold path once the pose moves
// far enough that the previous solution is a stale guess. See DEVIATIONS
// §9c for the speed table.
//
// Scope: proximityJacobian's outputs -- witness point, alpha, and
// d[witness;alpha]/dxi (the first-order IFT solve) -- warm-start cleanly.
// proximityContactJacobian's extra d(normal)/dxi does NOT: its frozen
// Hessian (hessianFrozenFull) needs the converged (s,z) exactly conically
// complementary, which a warm-started interior-point point does not
// reliably reach, so that call has no warm overload (verified in
// tests/test_warm_start.cpp -- witness/alpha/jacobian match cold to <=1e-5
// while d(normal)/dxi can diverge).
//
// Design (see DEVIATIONS.md for the reasoning):
//  - The handle is caller-owned, one per persistent contact pair -- mirrors
//    a physics engine's persistent contact manifold. Passing nullptr is
//    byte-for-byte the cold path; nothing about the default behaviour
//    changes.
//  - A trust-region gate on how far `g` moved from the cached pose: outside
//    it, fall back to a cold solve. The radius adapts -- grows after a
//    cheap warm solve, halves after a fallback -- so a trajectory passing
//    near an ill-conditioned configuration self-throttles and recovers.
//  - A warm solve whose KKT residuals lag (mu can dip below pdip_tol on a
//    warm path while x still settles) is rejected and redone cold, so a
//    warm answer is never lower quality than the cold one.
//  - Warm and cold converge to the same optimum at the same pdip_tol; the
//    only difference is iteration count. Verified in tests/test_warm_start.cpp.

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// Combined-SOCP dimensions for a (Shape1, Shape2) pair, read straight off
// the deduced problemMatrices return types -- mirrors combineProblemMatrices'
// own layout (nx = v1 + (v2 - 4); n_ort = n_ort1 + n_ort2).
template <class Shape1, class Shape2>
struct PairSocpDims {
    using PM1 = decltype(problemMatrices(std::declval<const Shape1&>(), std::declval<const Eigen::Matrix4d&>()));
    using PM2 = decltype(problemMatrices(std::declval<const Shape2&>(), std::declval<const Eigen::Matrix4d&>()));
    static constexpr int n_ort1 = decltype(std::declval<PM1&>().G_ort)::RowsAtCompileTime;
    static constexpr int v1 = decltype(std::declval<PM1&>().G_ort)::ColsAtCompileTime;
    static constexpr int n_soc1 = decltype(std::declval<PM1&>().G_soc)::RowsAtCompileTime;
    static constexpr int n_ort2 = decltype(std::declval<PM2&>().G_ort)::RowsAtCompileTime;
    static constexpr int v2 = decltype(std::declval<PM2&>().G_ort)::ColsAtCompileTime;
    static constexpr int n_soc2 = decltype(std::declval<PM2&>().G_soc)::RowsAtCompileTime;
    static constexpr int n_ort = n_ort1 + n_ort2;
    static constexpr int nx = v1 + (v2 - 4);
};

// Tuning constants for the warm path. Deliberately conservative: a warm
// solve that needs more than kMaxIters iterations, or whose KKT residuals
// exceed kResidTolMul * pdip_tol, is treated as a bad hint and redone cold,
// so these only affect speed, never the answer.
struct WarmStartConfig {
    static constexpr int kMaxIters = 16;        // cap on the warm solve before falling back
    static constexpr double kResidTolMul = 1e4; // reject a warm solve whose KKT residual > this * pdip_tol
    static constexpr int kCheapIters = 4;       // a warm solve this quick grows the trust radius
    static constexpr double kRhoInit = 0.02;    // initial trust radius on the pose-move metric
    static constexpr double kRhoMin = 1e-3;     // never gate below this
    static constexpr double kRhoMax = 0.75;     // never grow above this (grows toward it on a smooth path)
    static constexpr double kRhoGrow = 1.5;     // multiplier after a cheap warm solve
    static constexpr double kRhoShrink = 0.5;   // multiplier after a fallback
};

// Per-pair warm-start handle. Fixed size (dims deduced from the shape
// types), trivially copyable, no allocation -- keep one per persistent
// contact pair (e.g. in a map keyed by broadphase pair id) and pass its
// address to proximityJacobian every step.
//
// If shape1's or shape2's parameters / r_offset / R_offset are mutated
// mid-session, call reset() -- the handle assumes the pair is stable
// (it caches body 1's pose-independent problem matrices, and the previous
// solution).
template <class Shape1, class Shape2>
struct ContactWarmState {
    using D = PairSocpDims<Shape1, Shape2>;

    Eigen::Matrix4d g_ref = Eigen::Matrix4d::Identity(); // pose the cached x/s/z solves
    DecisionVec<D::nx> x = DecisionVec<D::nx>::Zero();
    StackVec<D::n_ort, D::n_soc1, D::n_soc2> s = StackVec<D::n_ort, D::n_soc1, D::n_soc2>::Zero();
    StackVec<D::n_ort, D::n_soc1, D::n_soc2> z = StackVec<D::n_ort, D::n_soc1, D::n_soc2>::Zero();
    double mu = 0.0;                       // s.z/degree at that solve
    double rho = WarmStartConfig::kRhoInit; // adaptive trust radius
    bool valid = false;                    // false until the first converged solve fills this in

    // Body 1 sits at the pair's reference frame (g = Identity), so its
    // (G_ort, h_ort, G_soc, h_soc) don't depend on the query pose -- built
    // once here and reused every call.
    ProblemMats<D::n_ort1, D::n_soc1, D::v1> P1;
    bool P1_valid = false;

    void reset() { *this = ContactWarmState{}; }
};

// A scale-free "how far did the pose move" metric between two SE(3) poses,
// used only for the trust-region gate (an exact twist norm via se3::Log
// isn't needed to decide "small enough"). For a small relative motion this
// is ~ sqrt( 2*angle^2 + ||translation||^2 ).
inline double poseMoveMetric(const Eigen::Matrix4d& a, const Eigen::Matrix4d& b) {
    const Eigen::Matrix4d d = se3::SE3Inverse(a) * b;
    const double rot = (d.block<3, 3>(0, 0) - Eigen::Matrix3d::Identity()).norm();
    const double trans = d.block<3, 1>(0, 3).norm();
    return std::sqrt(rot * rot + trans * trans);
}

} // namespace dcolpp::socp
