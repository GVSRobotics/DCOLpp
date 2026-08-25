#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
// Source: src/proximity_gradient.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Cheaper alternative to proximityJacobian when only d(alpha)/dxi is needed,
// not the full 4x6 Jacobian: by the envelope theorem, at a KKT point this
// equals -q^T*z (proximityGradientAnalytic, analytic_derivatives.hpp), q
// from Stage 1-3.

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/proximity.hpp"

namespace dcolpp::socp {

struct ProximityGradientResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero(); // d(alpha)/dxi, xi=[w;v]
    int iters = 0;
    bool converged = false;
};

template <typename Shape1, typename Shape2>
ProximityGradientResult proximityGradient(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(shape1, shape2, g, combined.c, combined.G, combined.h, opt);

    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
    constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

    ProximityGradientResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.grad = proximityGradientAnalytic<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
            shape1, shape2, sol.x, sol.z, g);
    }
    return res;
}

// Contact normal at the witness point, in the pair's reference (shape-1)
// frame. Per Le Cleac'h et al., "Single-Level Differentiable Contact
// Simulation" (RAL 2023), Eq. 14: n proportional to (d phi/dx)^T, phi =
// alpha - 1 the optimization-based SDF -- the same envelope-theorem
// derivative `grad` already carries, w.r.t. a 6-dof relative-pose twist
// instead of a raw 3-dof world position.
//
// `grad`'s translational block (tail<3>(), xi=[w;v]) is d(alpha)/dv, v
// being the LOCAL (body-2-frame) direction `retract` perturbs by -- not the
// reference frame. A local step dv moves the reference-frame position by
// g.linear()*dv, so d/d(p_ref) = g.linear() * d/dv. `g` must be the pose
// passed to `proximityGradient` that produced `r`.
//
// Zero vector if `r` didn't converge or grad's translational block
// vanishes (degenerate configuration) -- check `r.converged` first.
inline Eigen::Vector3d contactNormal(const ProximityGradientResult& r, const Eigen::Matrix4d& g) {
    return (g.block<3, 3>(0, 0) * r.grad.template tail<3>().transpose()).normalized();
}

} // namespace dcolpp::socp
