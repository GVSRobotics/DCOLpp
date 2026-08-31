#pragma once
// dcolpp::socp -- the proximity query: minimum uniform scaling `alpha`
// (< 1 penetrating, == 1 touching, > 1 separated) and the witness point,
// both in shape 1's frame, and their derivatives w.r.t. shape 2's relative
// pose (analytic_derivatives.hpp).
//
//   proximity          -- alpha, witness (one SOCP solve)
//   alphaGradient      -- + d(alpha)/dxi   (O(1) envelope gradient)
//   proximityJacobian  -- + d[witness;alpha]/dxi   (the full IFT solve)
//
// Each takes an optional ContactWarmState* for temporally-continuous
// callers (warm_start.hpp); nullptr is the cold path. The contact normal
// and its Jacobian, and degenerate-contact handling, live in contact.hpp
// on top of these.

#include <algorithm>

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/socp_init.hpp"
#include "dcolpp/socp/solver.hpp"
#include "dcolpp/socp/warm_start.hpp"

namespace dcolpp::socp {

// --- forward-solve orchestration -------------------------------------------
// Turns (shapes, pose, combined c/G/h) into a converged SocpResult: picks
// the cold init strategy, and -- given a ContactWarmState (warm_start.hpp) --
// tries a warm solve and falls back to cold when it doesn't pan out. The
// SocpInit seeds themselves are in socp_init.hpp.

// Cold solve: pick opt.init_strategy, seed, solve. solveSocp never has to
// know which strategy produced the hint.
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

// Body 1's problem matrices (pose-independent, g = Identity). Returned from
// the warm handle's cache when present (filled on first use), else built
// into `local` -- which must outlive the returned reference.
template <typename Shape1, typename Shape2, typename P1T>
const P1T& cachedBody1Matrices(const Shape1& shape1, ContactWarmState<Shape1, Shape2>* warm, P1T& local) {
    if (warm != nullptr) {
        if (!warm->P1_valid) {
            warm->P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
            warm->P1_valid = true;
        }
        return warm->P1;
    }
    local = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    return local;
}

// Which warm seed a query needs. Standard (warmStartInit): a cheap per-block
// lift back into the interior; s o z ends up non-uniform, fine for the value
// and the first-order Jacobian. Central (warmStartInitCentral): a uniform
// lift preserving s o z proportional to e -- required when the frozen Hessian
// (d(normal)/dxi) is downstream.
enum class WarmSeed { Standard, Central };

// One forward solve for a proximity-family query, with optional warm-start.
// With a valid handle and the pose inside its trust radius: seed from the
// previous (x, s, z), solve (iteration-capped), and keep the result if its
// KKT residuals pass resid_tol_mul * pdip_tol; otherwise redo cold. Adapts
// warm->rho, and on any converged solve writes (g, x, s, z, mu) back to the
// handle. nullptr -> plain cold solve.
template <int n_ort, int n_soc1, int n_soc2, int nx, typename Shape1, typename Shape2>
SocpResult<n_ort, n_soc1, n_soc2, nx> solveForQuery(const Shape1& shape1, const Shape2& shape2,
                                                     const Eigen::Matrix4d& g, const DecisionVec<nx>& c,
                                                     const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                     const StackVec<n_ort, n_soc1, n_soc2>& h, const SocpOptions& opt,
                                                     ContactWarmState<Shape1, Shape2>* warm, WarmSeed seed,
                                                     double resid_tol_mul) {
    SocpResult<n_ort, n_soc1, n_soc2, nx> sol;
    bool warm_used = false;
    const double pose_move = (warm != nullptr && warm->valid) ? poseMoveMetric(warm->g_ref, g) : 0.0;

    if (warm != nullptr && warm->valid && pose_move <= warm->rho) {
        const auto wi = (seed == WarmSeed::Central)
                            ? warmStartInitCentral<n_ort, n_soc1, n_soc2, nx>(c, G, h, warm->x, warm->z)
                            : warmStartInit<n_ort, n_soc1, n_soc2, nx>(c, G, h, warm->x, warm->z, pose_move);
        SocpOptions wopt = opt;
        wopt.max_iters = std::min(opt.max_iters, WarmStartConfig::kMaxIters);
        sol = solveSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h, wopt, &wi);
        if (sol.converged) {
            const double res_tol = resid_tol_mul * opt.pdip_tol;
            const double rx = (G.transpose() * sol.z + c).norm();
            const double rz = (sol.s + G * sol.x - h).norm();
            warm_used = (rx <= res_tol && rz <= res_tol);
        }
    }
    if (!warm_used) {
        sol = solveProximitySocp<n_ort, n_soc1, n_soc2, nx>(shape1, shape2, g, c, G, h, opt);
        if (warm != nullptr && warm->valid)
            warm->rho = std::max(WarmStartConfig::kRhoShrink * warm->rho, WarmStartConfig::kRhoMin);
    } else if (warm != nullptr && sol.iters <= WarmStartConfig::kCheapIters) {
        warm->rho = std::min(WarmStartConfig::kRhoGrow * warm->rho, WarmStartConfig::kRhoMax);
    }

    if (warm != nullptr) {
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

struct ProximityResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero(); // reference (shape-1) frame
    int iters = 0;
    bool converged = false;
};

template <typename Shape1, typename Shape2>
ProximityResult proximity(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                           const SocpOptions& opt = SocpOptions{},
                           ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    decltype(problemMatrices(shape1, Eigen::Matrix4d::Identity())) P1_local;
    const auto& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveForQuery<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard,
        WarmStartConfig::kResidTolMul);

    ProximityResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

struct AlphaGradientResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero(); // d(alpha)/dxi, xi=[w;v]
    int iters = 0;
    bool converged = false;
};

// Solve + d(alpha)/dxi only -- the envelope-theorem gradient
// (computeProximityGradient), O(1) after the solve. Cheaper than
// proximityJacobian, which also returns the witness point's rows.
template <typename Shape1, typename Shape2>
AlphaGradientResult alphaGradient(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                   const SocpOptions& opt = SocpOptions{},
                                   ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    decltype(problemMatrices(shape1, Eigen::Matrix4d::Identity())) P1_local;
    const auto& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveForQuery<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard,
        WarmStartConfig::kResidTolMul);

    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
    constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

    AlphaGradientResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.grad = computeProximityGradient<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
            shape1, shape2, sol.x, sol.z, g);
    }
    return res;
}

// d[p;alpha]/dxi (4x6) via the implicit function theorem at the converged
// (x,s,z) (analytic_derivatives.hpp). Shape widths (v1,v2) and shape 1's ORT
// row count are read off the deduced problemMatrices types, so either shape
// may carry any decision-vector width.
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

// As above but takes G directly -- proximityJacobian already built it for
// the forward solve.
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

// The optional `warm` handle warm-starts the SOCP for temporally-continuous
// queries (warm_start.hpp); nullptr is exactly the cold path. This path's
// outputs tolerate the looser (s,z) a fast warm solve reaches, so it accepts
// at mu < pdip_tol -- unlike proximityContactJacobian, whose d(normal)/dxi
// needs a central-path point.
template <typename Shape1, typename Shape2>
ProximityJacobianResult proximityJacobian(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{},
                                           ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    decltype(problemMatrices(shape1, Eigen::Matrix4d::Identity())) P1_local;
    const auto& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveForQuery<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard,
        WarmStartConfig::kResidTolMul);

    ProximityJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.jacobian = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);
    }
    return res;
}

} // namespace dcolpp::socp
