#pragma once
// dcolpp::socp -- ported from DifferentiableCollisions.jl
//
// The three quantities most callers actually want out of a proximity query
// -- witness point, alpha, contact normal -- bundled into one call, with a
// second variant that also returns their Jacobians w.r.t. the 6-dof
// relative-pose twist xi=[w;v]. Everything here is a thin composition of
// proximity.hpp/proximity_gradient.hpp/analytic_derivatives.hpp
// /contact_degeneracy.hpp/contact_manifold.hpp; no new derivative math
// lives in this file.
//
// proximityContact: one SOCP solve + the cheap envelope-theorem gradient
// (proximityGradientAnalytic, O(1) after the solve -- no IFT elimination).
// Use this when Jacobians aren't needed; it's strictly cheaper than
// proximityContactJacobian.
//
// proximityContactJacobian: one SOCP solve + the full IFT sensitivity
// (diffSocp, for d[witness;alpha]/dxi) + the Hessian-based normal
// sensitivity (contactNormalJacobian, for d(normal)/dxi). This is the
// expensive path -- it needs proximityHessianAnalytic internally. Optionally
// also computes ContactDegeneracy's diagnostics and/or ContactManifold's
// multi-point witness set (see contact_degeneracy.hpp/contact_manifold.hpp)
// when a degenerate contact makes a single witness point/Jacobian
// insufficient -- both opt-in (SocpOptions::compute_degeneracy_info/
// compute_contact_manifold), since neither is free.

#include <Eigen/Dense>
#include <vector>

#include "dcolpp/socp/contact_degeneracy.hpp"
#include "dcolpp/socp/contact_manifold.hpp"
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

    // Degeneracy diagnostics (ContactDegeneracy, contact_degeneracy.hpp) --
    // only computed when opt.compute_degeneracy_info is true (measured at
    // ~10-20% of this function's total cost, not negligible, so it's
    // opt-in, off by default). When not computed, the dims stay at the -1
    // sentinel and the booleans default to true -- true here means "not
    // checked", not "confirmed valid"; check the dims for -1 to tell the
    // difference. alpha's own row of `jacobian` is always well-defined
    // regardless of witness_jacobian_valid -- it's the witness-point ROWS
    // specifically that go bad when witness_jacobian_valid is false.
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;

    // Multi-point ContactManifold (contact_manifold.hpp) -- only populated
    // when opt.compute_contact_manifold is true (strictly more work than
    // compute_degeneracy_info alone; implies it). Empty when not requested,
    // 1/2/opt.contact_manifold_points points when requested, matching
    // contact_manifold_dim == 0/1/2.
    std::vector<Eigen::Vector3d> contact_manifold_points;
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

        if constexpr (IsStrictlyConvex<Shape1>::value || IsStrictlyConvex<Shape2>::value) {
            // Provably contact_manifold_dim==0 and normal_cone_dim==0
            // whenever either shape is strictly convex (see primitives.hpp:
            // IsStrictlyConvex) -- skip the SVD/LU machinery ENTIRELY, not
            // just cheapen it, whenever the caller asked for either
            // diagnostic. Only Sphere/Ellipsoid qualify among these 7
            // shapes; anything else falls through to the real computation
            // below, unchanged.
            if (opt.compute_contact_manifold || opt.compute_degeneracy_info) {
                res.contact_manifold_dim = 0;
                res.normal_cone_dim = 0;
                res.witness_jacobian_valid = true;
                res.normal_jacobian_valid = true;
                if (opt.compute_contact_manifold) {
                    res.contact_manifold_points.push_back(res.witness_point);
                }
            }
        } else if (opt.compute_contact_manifold) {
            const auto cm = contactManifold<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
                sol.x, sol.s, sol.z, combined.G, opt.contact_manifold_points);
            res.contact_manifold_dim = cm.contact_manifold_dim;
            res.normal_cone_dim = cm.normal_cone_dim;
            res.witness_jacobian_valid = cm.witness_jacobian_valid;
            res.normal_jacobian_valid = cm.normal_jacobian_valid;
            res.contact_manifold_points = cm.witness_points;
        } else if (opt.compute_degeneracy_info) {
            const auto degen = contactDegeneracy<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
                sol.s, sol.z, combined.G);
            res.contact_manifold_dim = degen.contact_manifold_dim;
            res.normal_cone_dim = degen.normal_cone_dim;
            res.witness_jacobian_valid = degen.witness_jacobian_valid;
            res.normal_jacobian_valid = degen.normal_jacobian_valid;
        }
    }
    return res;
}

} // namespace dcolpp::socp
