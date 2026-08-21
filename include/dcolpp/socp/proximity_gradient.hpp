#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/proximity_gradient.jl (Kevin Tracy, MIT License). See
// NOTICE.md.
//
// A cheaper alternative to proximityJacobian (proximity.hpp) when only the
// gradient of `alpha` itself is needed, not the full 4x6 Jacobian of
// [witness point; alpha]: by the envelope theorem, d(alpha)/dxi at a KKT
// point equals the partial derivative (holding x,s,z fixed at their
// converged values) of the Lagrangian's constraint term z'(Gx - h) w.r.t.
// xi -- a single scalar Dual6 evaluation, no NT-scaling linear solve at all.

#include "dcolpp/dual6.hpp"
#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"

namespace dcolpp::socp {

template <typename T, typename Shape1, typename Shape2, int NX, int NS>
T lagConPart(const Shape1& shape1, const Shape2& shape2, const TVec<NX, double>& x, const TVec<NS, double>& z,
             const TMat<4, 4, T>& g2) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = castProblemMats<T>(problemMatrices<double>(shape1, I4));
    const auto P2 = problemMatrices<T>(shape2, g2);
    const auto combined = combineProblemMatrices<T>(P1, P2);

    const TVec<NX, T> xT = x.template cast<T>();
    const TVec<NS, T> zT = z.template cast<T>();
    return zT.dot(combined.G * xT - combined.h);
}

template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 1, 6> objValGrad(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                        const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    constexpr int ns = n_ort + n_soc1 + n_soc2;
    const TMat<4, 4, Dual6> g2D = se3::retract<Dual6>(g0, se3::seedTwist());
    const Dual6 val = lagConPart<Dual6, Shape1, Shape2, nx, ns>(shape1, shape2, x, z, g2D);
    return val.grad().transpose();
}

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
    const auto P1 = problemMatrices<double>(shape1, I4);
    const auto P2 = problemMatrices<double>(shape2, g);
    const auto combined = combineProblemMatrices<double>(P1, P2);

    const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        combined.c, combined.G, combined.h, opt);

    ProximityGradientResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.grad = objValGrad<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.z, g);
    }
    return res;
}

// Contact normal at the witness point, in the pair's reference (shape-1)
// frame -- matching `ProximityGradientResult::witness_point`'s frame. Per Le
// Cleac'h et al., "Single-Level Differentiable Contact Simulation" (RAL
// 2023), Eq. 14: n ∝ (∂φ/∂x)^T where φ = alpha - 1 is the optimization-based
// SDF and x is shape 2's position in the reference frame -- the same
// envelope-theorem derivative `grad` already carries, just w.r.t. a 6-dof
// relative-pose twist instead of a raw 3-dof world position.
//
// `grad`'s translational block (`.tail<3>()`, per xi=[w;v]) is d(alpha)/dv,
// where v is the LOCAL (body-2-frame) translation direction `retract`
// perturbs by -- not the reference frame. Since a local step dv moves the
// reference-frame position by g.linear()*dv, d/d(p_ref) = g.linear() *
// d/dv. `g` must be the same pose passed to `proximityGradient` that
// produced `r`.
//
// Undefined (zero vector) if `r` didn't converge or if the gradient's
// translational block vanishes (degenerate configuration); callers should
// check `r.converged` first.
inline Eigen::Vector3d contactNormal(const ProximityGradientResult& r, const Eigen::Matrix4d& g) {
    return (g.block<3, 3>(0, 0) * r.grad.template tail<3>().transpose()).normalized();
}

} // namespace dcolpp::socp
