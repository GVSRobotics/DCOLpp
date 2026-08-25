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
#include "dcolpp/socp/geometric_init.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// Shared by proximity()/proximityJacobian()/proximityGradient(): branches
// on opt.init_strategy so solveSocp itself only ever sees a plain
// (c,G,h,opt) call or one with an explicit init_hint -- never has to know
// which strategy produced it.
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
    const auto sens = diffSocpSensitivityAnalyticAuto<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, x, s, z, g0);
    return sens.dx.template topRows<4>();
}

// Same as diffSocp above, but takes G directly (see
// diffSocpSensitivityAnalyticWithG); used by proximityJacobian, which
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
    const auto sens = diffSocpSensitivityAnalyticAutoWithG<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
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

template <typename Shape1, typename Shape2>
ProximityJacobianResult proximityJacobian(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(shape1, shape2, g, combined.c, combined.G, combined.h, opt);

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
