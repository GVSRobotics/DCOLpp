#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/proximity.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Forward proximity query: given two primitives and the relative SE(3) pose
// `g` placing shape 2 into shape 1's frame (shape 1 sits at Identity, per
// the pair convention used throughout DCOL++; see problem_matrices.hpp),
// solve the SOCP and report the minimum uniform scaling `alpha` (alpha < 1:
// penetrating, alpha == 1: touching, alpha > 1: separated) and the witness
// point, both in shape 1's / the pair's reference frame.
//
// Differentiation (proximity_jacobian / proximity_gradient, porting
// kkt_R/diff_socp) is added in Phase 3.

#include <Eigen/Dense>

#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

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

    const auto P1 = problemMatrices<double>(shape1, I4);
    const auto P2 = problemMatrices<double>(shape2, g);
    const auto combined = combineProblemMatrices<double>(P1, P2);

    const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        combined.c, combined.G, combined.h, opt);

    ProximityResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

} // namespace dcolpp::socp
