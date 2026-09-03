#pragma once
// dcolpp::socp::runtime_poly -- contact query for PolytopeX (shape 1) vs a curved
// primitive (shape 2). Mirrors contact.hpp; witnesses via the dim-free
// dcolpp::socp::contactWitnesses.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp" // ContactWitnesses / contactWitnesses
#include "dcolpp/socp/runtime_poly/contact.hpp"
#include "dcolpp/socp/runtime_poly/contact_manifold_prim.hpp"
#include "dcolpp/socp/runtime_poly/proximity_prim.hpp"

namespace dcolpp::socp::runtime_poly {

template <typename Prim>
ProximityContactResultX proximityContact(const PolytopeX& hull, const Prim& prim, const Eigen::Matrix4d& g,
                                         const SocpOptions& opt = SocpOptions{}) {
    const ProblemMatsX P1 = problemMatrices(hull, Eigen::Matrix4d::Identity());
    const auto P2 = dcolpp::socp::problemMatrices(prim, g);
    constexpr int V2 = decltype(P2.G_ort)::ColsAtCompileTime;
    constexpr int NS2 = decltype(P2.G_soc)::RowsAtCompileTime;
    const auto cp = combineHullPrim(P1, P2);
    const auto sol = solveProximitySocpPrim<V2, NS2>(hull.bounding_sphere, prim, g, cp, opt);

    ProximityContactResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        const Eigen::Vector3d r_org = g.block<3, 1>(0, 3);
        const Eigen::Matrix<double, 1, 6> grad =
            proximityGradientPrim<V2, NS2>(prim, g, sol.x, sol.z, cp.n_ort1);
        res.normal = (g.block<3, 3>(0, 0) * grad.template tail<3>().transpose()).normalized();
        const ContactWitnesses w = contactWitnesses(res.witness_point, res.alpha, r_org, /*plane_body1=*/false);
        res.witness_body1 = w.body1;
        res.witness_body2 = w.body2;
        res.gap = w.gap;

        if (opt.compute_contact_manifold) {
            const auto cm = contactManifoldPrim<V2, NS2>(sol.x, sol.s, sol.z, cp.G, cp.n_ort, opt.contact_manifold_points);
            res.contact_manifold_dim = cm.contact_manifold_dim;
            res.normal_cone_dim = cm.normal_cone_dim;
            res.witness_jacobian_valid = cm.witness_jacobian_valid;
            res.normal_jacobian_valid = cm.normal_jacobian_valid;
            res.contact_manifold_points = cm.witness_points;
        } else if (opt.compute_degeneracy_info) {
            const auto d = contactDegeneracyPrim<V2, NS2>(sol.s, sol.z, cp.G, cp.n_ort);
            res.contact_manifold_dim = d.contact_manifold_dim;
            res.normal_cone_dim = d.normal_cone_dim;
            res.witness_jacobian_valid = d.witness_jacobian_valid;
            res.normal_jacobian_valid = d.normal_jacobian_valid;
        }
        for (const auto& p : res.contact_manifold_points)
            res.contact_manifold_witnesses.push_back(contactWitnesses(p, res.alpha, r_org, false));
    }
    return res;
}

template <typename Prim>
ProximityContactJacobianResultX proximityContactJacobian(const PolytopeX& hull, const Prim& prim,
                                                         const Eigen::Matrix4d& g,
                                                         const SocpOptions& opt = SocpOptions{}) {
    const ProblemMatsX P1 = problemMatrices(hull, Eigen::Matrix4d::Identity());
    const auto P2 = dcolpp::socp::problemMatrices(prim, g);
    constexpr int V2 = decltype(P2.G_ort)::ColsAtCompileTime;
    constexpr int NS2 = decltype(P2.G_soc)::RowsAtCompileTime;
    const auto cp = combineHullPrim(P1, P2);
    const auto sol = solveProximitySocpPrim<V2, NS2>(hull.bounding_sphere, prim, g, cp, opt);

    ProximityContactJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        const auto bundle = contactJacobianBundlePrim<V2, NS2>(prim, g, sol.x, sol.s, sol.z, cp.G, cp.n_ort1);
        res.jacobian = bundle.jacobian;
        res.normal = (g.block<3, 3>(0, 0) * bundle.grad.template tail<3>().transpose()).normalized();
        res.normal_jacobian = bundle.normal_jacobian;

        const Eigen::Vector3d r_org = g.block<3, 1>(0, 3);
        const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g, Eigen::Vector3d::Zero());
        const ContactWitnesses w =
            contactWitnesses(res.witness_point, res.alpha, r_org, res.jacobian, dr_dxi, /*plane_body1=*/false);
        res.witness_body1 = w.body1;
        res.witness_body2 = w.body2;
        res.gap = w.gap;
        res.witness_body1_jacobian = w.body1_jacobian;
        res.witness_body2_jacobian = w.body2_jacobian;
        res.gap_jacobian = w.gap_jacobian;

        if (opt.compute_contact_manifold) {
            const auto cm = contactManifoldPrim<V2, NS2>(sol.x, sol.s, sol.z, cp.G, cp.n_ort, opt.contact_manifold_points);
            res.contact_manifold_dim = cm.contact_manifold_dim;
            res.normal_cone_dim = cm.normal_cone_dim;
            res.witness_jacobian_valid = cm.witness_jacobian_valid;
            res.normal_jacobian_valid = cm.normal_jacobian_valid;
            res.contact_manifold_points = cm.witness_points;
            for (const auto& p : cm.witness_points) {
                Eigen::Matrix<double, V2, 1> xp = sol.x;
                xp.template head<3>() = p;
                res.contact_manifold_point_jacobians.push_back(
                    diffSocpPrim<V2, NS2>(prim, g, xp, sol.s, sol.z, cp.G, cp.n_ort1));
                res.contact_manifold_point_normal_jacobians.push_back(res.normal_jacobian);
            }
        } else if (opt.compute_degeneracy_info) {
            const auto d = contactDegeneracyPrim<V2, NS2>(sol.s, sol.z, cp.G, cp.n_ort);
            res.contact_manifold_dim = d.contact_manifold_dim;
            res.normal_cone_dim = d.normal_cone_dim;
            res.witness_jacobian_valid = d.witness_jacobian_valid;
            res.normal_jacobian_valid = d.normal_jacobian_valid;
        }
        for (size_t i = 0; i < res.contact_manifold_points.size(); ++i)
            res.contact_manifold_witnesses.push_back(contactWitnesses(
                res.contact_manifold_points[i], res.alpha, r_org, res.contact_manifold_point_jacobians[i], dr_dxi,
                false));
    }
    return res;
}

} // namespace dcolpp::socp::runtime_poly
