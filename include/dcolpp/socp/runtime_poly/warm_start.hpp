#pragma once
// dcolpp::socp::runtime_poly -- per-pair warm-start state + orchestration for
// temporally-continuous PolytopeX proximity queries (a physics step, a
// traj-opt inner loop). Pure-ORT pairs (PolytopeX-PolytopeX, Plane-PolytopeX):
// runtime-row-count copy of socp_init.hpp's warmStartInit* + proximity.hpp's
// solveForQuery. poseMoveMetric / WarmStartConfig are reused from
// dcolpp/socp/warm_start.hpp unchanged.
//
// hull-vs-curved-primitive warm-start (SOC block) is a follow-up; those queries
// run cold via proximity_prim.hpp / contact_prim.hpp.

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "dcolpp/socp/warm_start.hpp" // poseMoveMetric, WarmStartConfig
#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"
#include "dcolpp/socp/runtime_poly/solver.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::poseMoveMetric;
using dcolpp::socp::WarmStartConfig;

// Per-pair handle: keep one per persistent contact pair, pass its address to
// every step's query. Caches body 1's pose-independent problem matrices and
// the previous converged (x, s, z). call reset() if a shape's params change.
struct ContactWarmStateX {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix4d g_ref = Eigen::Matrix4d::Identity();
    Eigen::Vector4d x = Eigen::Vector4d::Zero();
    StackVecX s;
    StackVecX z;
    double mu = 0.0;
    double rho = WarmStartConfig::kRhoInit;
    bool valid = false;

    ProblemMatsX P1;
    bool P1_valid = false;

    void reset() { *this = ContactWarmStateX{}; }
};

// socp_init.hpp warmRecenter / liftToMargin, ORT-only.
inline StackVecX warmRecenter(const StackVecX& r, double ort_floor) {
    StackVecX out = r;
    for (int i = 0; i < out.size(); ++i)
        if (out(i) < ort_floor) out(i) = ort_floor;
    return out;
}
inline StackVecX liftToMargin(const StackVecX& r, double target) {
    const double m = r.minCoeff();
    if (m >= target) return r;
    return (r.array() + (target - m)).matrix();
}

inline SocpInitX warmStartInit(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                               const Eigen::Vector4d& x_prev, const StackVecX& z_prev, double pose_move) {
    constexpr double kTinyMove = 1e-6;
    if (pose_move < kTinyMove) {
        return {x_prev, warmRecenter(StackVecX(h - G * x_prev), 1e-12), warmRecenter(z_prev, 1e-12)};
    }
    const StackVecX s0 = warmRecenter(StackVecX(h - G * x_prev), 1e-7);
    SmallLLT<4> F;
    F.compute(gram4(G));
    const Eigen::Vector4d w = F.solve(Eigen::Vector4d(-c - G.transpose() * z_prev));
    return {x_prev, s0, warmRecenter(StackVecX(z_prev + G * w), 1e-7)};
}

inline SocpInitX warmStartInitCentral(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                                      const Eigen::Vector4d& x_prev, const StackVecX& z_prev,
                                      double interior_margin = 1e-9) {
    const StackVecX s0 = liftToMargin(StackVecX(h - G * x_prev), interior_margin);
    SmallLLT<4> F;
    F.compute(gram4(G));
    const Eigen::Vector4d w = F.solve(Eigen::Vector4d(-c - G.transpose() * z_prev));
    return {x_prev, s0, liftToMargin(StackVecX(z_prev + G * w), interior_margin)};
}

enum class WarmSeed { Standard, Central };

// proximity.hpp solveForQuery, ORT-only. `coldSolve` runs the geometric cold
// path for this pair kind (hull-hull vs plane-hull differ only there).
template <typename ColdSolveFn>
SocpResultX solveForQuery(const Eigen::Vector4d& c, const ConstraintMatX<4>& G, const StackVecX& h,
                          const SocpOptions& opt, ContactWarmStateX* warm, WarmSeed seed, double resid_tol_mul,
                          const Eigen::Matrix4d& g, ColdSolveFn&& coldSolve) {
    SocpResultX sol;
    bool warm_used = false;
    const double pose_move = (warm && warm->valid) ? poseMoveMetric(warm->g_ref, g) : 0.0;

    if (warm && warm->valid && pose_move <= warm->rho) {
        const SocpInitX wi = (seed == WarmSeed::Central)
                                 ? warmStartInitCentral(c, G, h, warm->x, warm->z)
                                 : warmStartInit(c, G, h, warm->x, warm->z, pose_move);
        SocpOptions wopt = opt;
        wopt.max_iters = std::min(opt.max_iters, WarmStartConfig::kMaxIters);
        sol = solveSocp(c, G, h, wopt, &wi);
        if (sol.converged) {
            const double res_tol = resid_tol_mul * opt.pdip_tol;
            const double rx = (G.transpose() * sol.z + c).norm();
            const double rz = (sol.s + G * sol.x - h).norm();
            warm_used = (rx <= res_tol && rz <= res_tol);
        }
    }
    if (!warm_used) {
        sol = coldSolve();
        if (warm && warm->valid)
            warm->rho = std::max(WarmStartConfig::kRhoShrink * warm->rho, WarmStartConfig::kRhoMin);
    } else if (warm && sol.iters <= WarmStartConfig::kCheapIters) {
        warm->rho = std::min(WarmStartConfig::kRhoGrow * warm->rho, WarmStartConfig::kRhoMax);
    }
    if (warm) {
        if (sol.converged) {
            warm->g_ref = g;
            warm->x = sol.x;
            warm->s = sol.s;
            warm->z = sol.z;
            warm->mu = sol.mu;
            warm->valid = true;
        } else {
            warm->valid = false;
        }
    }
    return sol;
}

// Body 1's problem matrices (g = Identity), from the warm cache when present.
inline const ProblemMatsX& cachedBody1Matrices(const PolytopeX& shape1, ContactWarmStateX* warm, ProblemMatsX& local) {
    if (warm) {
        if (!warm->P1_valid) {
            warm->P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
            warm->P1_valid = true;
        }
        return warm->P1;
    }
    local = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    return local;
}
inline const ProblemMatsX& cachedBody1Matrices(const Plane& plane, ContactWarmStateX* warm, ProblemMatsX& local) {
    if (warm) {
        if (!warm->P1_valid) {
            warm->P1 = planeProblemMatrix(plane);
            warm->P1_valid = true;
        }
        return warm->P1;
    }
    local = planeProblemMatrix(plane);
    return local;
}

} // namespace dcolpp::socp::runtime_poly
