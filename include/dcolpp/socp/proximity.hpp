#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
// Source: src/proximity.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Forward proximity query and its Jacobian: solve the SOCP for the minimum
// uniform scaling `alpha` (alpha < 1: penetrating, == 1: touching, > 1:
// separated) and witness point, both in shape 1's frame; differentiate
// w.r.t. shape 2's relative pose via analytic_derivatives.hpp.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/socp_init.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"
#include "dcolpp/socp/warm_start.hpp"

namespace dcolpp::socp {

// Shared by proximity()/proximityJacobian()/proximityGradient(): branches
// on opt.init_strategy so solveSocp itself only ever sees a plain
// (c,G,h,opt) call or one with an explicit init_hint -- never has to know
// which strategy produced it.
//
// Surrogate scaling (iDCOL manuscript Sec. III.B) was tried here and
// reverted -- NOT shipped, despite an earlier from-first-principles
// verification of the underlying homogeneity property (x*(g)=k*x*(g_S) for
// a translation rescaled by 1/k) and a from-tools/bench_surrogate.cpp
// speed measurement that looked like a real ~37-44% win at wide dynamic
// range. That speed measurement is now known to be invalid: it only
// checked `.converged` and iteration counts, never compared against a
// solved-directly reference. Doing that comparison found the surrogate
// SOLVE ITSELF (not the mapping-back, which was independently verified
// exact) converging to a genuinely wrong point for some poses at extreme
// scale (Cone-Polytope, alpha~992: a witness-point component off by ~200x,
// confirmed via the completely standard generic-init path at
// pdip_tol=1e-12 too, so not specific to the geometric guess or to a
// tolerance that was too loose) -- something about the rescaled problem
// itself becomes newly ill-conditioned/degenerate for at least some
// shape/pose combinations in a way the original (unscaled) problem is not.
// Root cause not yet identified. Left here as a documented dead end so it
// isn't retried blindly with the same "it converges, it's faster" check
// that missed this the first time -- any future attempt MUST verify
// against a directly-solved reference, not just convergence+iterations.
template <int n_ort, int n_soc1, int n_soc2, int nx, typename Shape1, typename Shape2>
SocpResult<n_ort, n_soc1, n_soc2, nx> solveProximitySocp(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g, const DecisionVec<nx>& c,
                                                          const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                          const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                          const SocpOptions& opt) {
    if (opt.init_strategy == SocpInitStrategy::Geometric) {
        const auto x0 = geometricPrimalGuess(shape1, shape2, g);
        const auto init = initializeSocpFromGuess<n_ort, n_soc1, n_soc2, nx>(c, G, h, x0);
        return solveSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h, opt, &init);
    }
    return solveSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h, opt);
}

struct ProximityResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero(); // reference (shape-1) frame
    int iters = 0;
    bool converged = false;
};

template <typename Shape1, typename Shape2>
ProximityResult proximity(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                           const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(shape1, shape2, g, combined.c, combined.G, combined.h, opt);

    ProximityResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

// diff_socp: d[p;alpha]/dxi, a 4x6 Jacobian, via the implicit function
// theorem at the converged (x,s,z) -- dR/dxi from analytic_derivatives.hpp's
// Stage 1-3 chain rule for whichever shape shape2 is, any decision-vector
// width on either shape. Shape 1's/shape 2's widths (v1,v2) and shape 1's
// own ORT row count are read off their deduced problemMatrices types.
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 4, 6> diffSocp(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                      const StackVec<n_ort, n_soc1, n_soc2>& s,
                                      const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int v2 = nx - v1 + 4;
    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int n_ort2 = n_ort - n_ort1;
    const auto sens = computeSocpSensitivityAuto<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, x, s, z, g0);
    return sens.dx.template topRows<4>();
}

// Same as diffSocp above, but takes G directly (see
// computeSocpSensitivityWithG); used by proximityJacobian, which
// already built G for the forward solve.
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 4, 6> diffSocp(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                      const StackVec<n_ort, n_soc1, n_soc2>& s,
                                      const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0,
                                      const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G) {
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int v2 = nx - v1 + 4;
    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int n_ort2 = n_ort - n_ort1;
    const auto sens = computeSocpSensitivityAutoWithG<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, x, s, z, g0, G);
    return sens.dx.template topRows<4>();
}

struct ProximityJacobianResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero(); // rows [wx,wy,wz,alpha], cols xi=[w;v]
    int iters = 0;
    bool converged = false;
};

// The optional `warm` handle warm-starts the SOCP for temporally-
// continuous queries (a physics step, a traj-opt inner loop, a smooth
// sweep) -- see warm_start.hpp. Passing nullptr is byte-for-byte the cold
// path. proximityJacobian's outputs (witness, alpha, and d[witness;alpha]/
// dxi via the first-order IFT solve) all warm-start cleanly; the extra
// d(normal)/dxi that proximityContactJacobian returns does NOT (its frozen
// Hessian needs (s,z) exactly conically complementary, which a warm-
// started interior-point point does not reliably reach), so that one is
// deliberately left cold-only.
template <typename Shape1, typename Shape2>
ProximityJacobianResult proximityJacobian(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{},
                                           ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    // Body 1's problem matrices are pose-independent (built at g = Identity).
    // With a warm handle they're computed once and reused across the loop;
    // without one, freshly each call (the cold path is unchanged).
    using P1T = decltype(problemMatrices(shape1, I4));
    P1T P1_local;
    const P1T* P1 = &P1_local;
    if (warm != nullptr) {
        if (!warm->P1_valid) {
            warm->P1 = problemMatrices(shape1, I4);
            warm->P1_valid = true;
        }
        P1 = &warm->P1;
    } else {
        P1_local = problemMatrices(shape1, I4);
    }

    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(*P1, P2);

    SocpResult<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx> sol;
    bool warm_used = false;
    const double pose_move = (warm != nullptr && warm->valid) ? poseMoveMetric(warm->g_ref, g) : 0.0;
    if (warm != nullptr && warm->valid && pose_move <= warm->rho) {
        const auto wi = warmStartInit<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, warm->x, warm->z, pose_move);
        SocpOptions wopt = opt;
        wopt.max_iters = std::min(opt.max_iters, WarmStartConfig::kMaxIters);
        sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, wopt, &wi);
        if (sol.converged) {
            // solveSocp already gates its iteration-1 early return on strict
            // (pdip_tol) residuals; this is a looser 1e4*pdip_tol re-check on
            // the final iterate, deciding whether the warm result is good
            // enough to keep or we fall back to a cold solve.
            const double res_tol = WarmStartConfig::kResidTolMul * opt.pdip_tol;
            const double rx = (combined.G.transpose() * sol.z + combined.c).norm();
            const double rz = (sol.s + combined.G * sol.x - combined.h).norm();
            warm_used = (rx <= res_tol && rz <= res_tol);
        }
    }
    if (!warm_used) {
        sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, g, combined.c, combined.G, combined.h, opt);
        if (warm != nullptr && warm->valid)
            warm->rho = std::max(WarmStartConfig::kRhoShrink * warm->rho, WarmStartConfig::kRhoMin);
    } else if (warm != nullptr && sol.iters <= WarmStartConfig::kCheapIters) {
        warm->rho = std::min(WarmStartConfig::kRhoGrow * warm->rho, WarmStartConfig::kRhoMax);
    }

    ProximityJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.jacobian = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);
    }

    if (warm != nullptr) {
        if (res.converged) {
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
    return res;
}

} // namespace dcolpp::socp
