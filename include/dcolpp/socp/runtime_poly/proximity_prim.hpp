#pragma once
// dcolpp::socp::runtime_poly -- proximity / Jacobian for PolytopeX (shape 1) vs a
// curved primitive (shape 2: Sphere / Ellipsoid / Capsule / Cylinder / Cone /
// TruncatedCone). Mirrors proximity.hpp; the SOC block is handled by
// solver_soc.hpp and the sensitivity by analytic_derivatives_prim.hpp.

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/analytic_derivatives_prim.hpp"
#include "dcolpp/socp/runtime_poly/problem_matrices_prim.hpp"
#include "dcolpp/socp/runtime_poly/proximity.hpp" // ProximityResultX / *JacobianResultX structs

namespace dcolpp::socp::runtime_poly {

template <typename Prim>
ProximityResultX proximity(const PolytopeX& hull, const Prim& prim, const Eigen::Matrix4d& g,
                           const SocpOptions& opt = SocpOptions{}) {
    const ProblemMatsX P1 = problemMatrices(hull, Eigen::Matrix4d::Identity());
    const auto P2 = dcolpp::socp::problemMatrices(prim, g);
    constexpr int V2 = decltype(P2.G_ort)::ColsAtCompileTime;
    constexpr int NS2 = decltype(P2.G_soc)::RowsAtCompileTime;
    const auto cp = combineHullPrim(P1, P2);
    const auto sol = solveProximitySocpPrim<V2, NS2>(hull.bounding_sphere, prim, g, cp, opt);

    ProximityResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

template <typename Prim>
ProximityJacobianResultX proximityJacobian(const PolytopeX& hull, const Prim& prim, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{}) {
    const ProblemMatsX P1 = problemMatrices(hull, Eigen::Matrix4d::Identity());
    const auto P2 = dcolpp::socp::problemMatrices(prim, g);
    constexpr int V2 = decltype(P2.G_ort)::ColsAtCompileTime;
    constexpr int NS2 = decltype(P2.G_soc)::RowsAtCompileTime;
    const auto cp = combineHullPrim(P1, P2);
    const auto sol = solveProximitySocpPrim<V2, NS2>(hull.bounding_sphere, prim, g, cp, opt);

    ProximityJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged)
        res.jacobian = diffSocpPrim<V2, NS2>(prim, g, sol.x, sol.s, sol.z, cp.G, cp.n_ort1);
    return res;
}

} // namespace dcolpp::socp::runtime_poly
