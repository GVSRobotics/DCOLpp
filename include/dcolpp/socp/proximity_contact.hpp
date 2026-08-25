#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
//
// The three quantities most callers actually want out of a proximity query
// -- witness point, alpha, contact normal -- bundled into one call, with a
// second variant that also returns their Jacobians w.r.t. the 6-dof
// relative-pose twist xi=[w;v]. Everything here is a thin composition of
// proximity.hpp/proximity_gradient.hpp/analytic_derivatives.hpp; no new
// derivative math lives in this file.
//
// proximityContact: one SOCP solve + the cheap envelope-theorem gradient
// (proximityGradientAnalytic, O(1) after the solve -- no IFT elimination).
// Use this when Jacobians aren't needed; it's strictly cheaper than
// proximityContactJacobian.
//
// proximityContactJacobian: one SOCP solve + the full IFT sensitivity
// (diffSocp, for d[witness;alpha]/dxi) + the Hessian-based normal
// sensitivity (contactNormalJacobian, for d(normal)/dxi). This is the
// expensive path -- it needs proximityHessianAnalytic internally.

#include <Eigen/Dense>

#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_gradient.hpp"

namespace dcolpp::socp {

struct ProximityContactResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero(); // reference (shape-1) frame
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();        // reference (shape-1) frame, unit
    int iters = 0;
    bool converged = false;
};

// (shape1, shape2, g) -> witness point, alpha, normal. No Jacobians -- see
// proximityContactJacobian below for that.
template <typename Shape1, typename Shape2>
ProximityContactResult proximityContact(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                         const SocpOptions& opt = SocpOptions{}) {
    const auto gr = proximityGradient(shape1, shape2, g, opt);

    ProximityContactResult res;
    res.alpha = gr.alpha;
    res.witness_point = gr.witness_point;
    res.iters = gr.iters;
    res.converged = gr.converged;
    if (gr.converged) {
        res.normal = contactNormal(gr, g);
    }
    return res;
}

struct ProximityContactJacobianResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();        // d[witness;alpha]/dxi
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero(); // d(normal)/dxi
    int iters = 0;
    bool converged = false;
};

// (shape1, shape2, g) -> witness point, alpha, normal, and all three
// Jacobians w.r.t. xi=[w;v] (jacobian's rows are [wx,wy,wz,alpha];
// normal_jacobian's rows are the normal's [nx,ny,nz]). Strictly more work
// than proximityContact -- pulls in the full Hessian machinery for
// normal_jacobian.
template <typename Shape1, typename Shape2>
ProximityContactJacobianResult proximityContactJacobian(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g,
                                                          const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = problemMatrices(shape1, I4);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt);

    ProximityContactJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.jacobian = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);

        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

        const Eigen::Matrix<double, 1, 6> grad =
            proximityGradientAnalytic<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                shape1, shape2, sol.x, sol.z, g);
        res.normal = (g.block<3, 3>(0, 0) * grad.template tail<3>().transpose()).normalized();

        res.normal_jacobian =
            contactNormalJacobian<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
                shape1, shape2, sol.x, sol.s, sol.z, g);
    }
    return res;
}

} // namespace dcolpp::socp
