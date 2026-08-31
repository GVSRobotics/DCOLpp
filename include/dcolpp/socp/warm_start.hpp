#pragma once
// dcolpp::socp -- the per-pair warm-start STATE and trust-region policy for
// temporally-continuous proximity queries (a physics step, a traj-opt inner
// loop, a smooth sweep). 

#include <cmath>
#include <utility>

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/problem_matrices.hpp"

namespace dcolpp::socp {

// Combined-SOCP dimensions for a (Shape1, Shape2) pair, off the deduced
// problemMatrices return types (nx = v1 + (v2 - 4); n_ort = n_ort1 + n_ort2).
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

// Tuning constants for the warm path -- affect speed only: a warm solve
// outside these bounds is treated as a bad hint and redone cold.
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

// Per-pair warm-start handle: fixed size, trivially copyable, no allocation.
// Keep one per persistent contact pair (e.g. keyed by broadphase pair id)
// and pass its address to the query every step. It caches body 1's
// pose-independent problem matrices and the previous solution, so call
// reset() if either shape's parameters are mutated mid-session.
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

    // Body 1's problem matrices (pose-independent, g = Identity), built once.
    ProblemMats<D::n_ort1, D::n_soc1, D::v1> P1;
    bool P1_valid = false;

    void reset() { *this = ContactWarmState{}; }
};

// Scale-free pose-move metric for the trust-region gate (~ sqrt(2*angle^2 +
// ||translation||^2) for small relative motion; no exact twist norm needed).
inline double poseMoveMetric(const Eigen::Matrix4d& a, const Eigen::Matrix4d& b) {
    const Eigen::Matrix4d d = se3::SE3Inverse(a) * b;
    const double rot = (d.block<3, 3>(0, 0) - Eigen::Matrix3d::Identity()).norm();
    const double trans = d.block<3, 1>(0, 3).norm();
    return std::sqrt(rot * rot + trans * trans);
}

} // namespace dcolpp::socp
