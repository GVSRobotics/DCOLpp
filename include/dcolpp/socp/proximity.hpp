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

#include "dcolpp/dual6.hpp"
#include "dcolpp/se3.hpp"
#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
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

// -------------------------------------------------------------------------
// Differentiation: kkt_R / diff_socp / proximity_jacobian, ported from the
// same source file (src/proximity.jl). See proximity_gradient.hpp for the
// cheaper alpha-only gradient (src/proximity_gradient.jl).
//
// The KKT residual R(x,s,z; g2) = [c + G'z ; cone_product(h - Gx, z)],
// evaluated at the CONVERGED double (x,s,z) from `proximity()` but with
// shape 2's placement g2 templated on scalar type T. Differentiating this
// small, closed-form residual w.r.t. g2's local twist (via Dual6) and
// contracting through the implicit function theorem at fixed (x,s,z) is
// the whole trick that makes this differentiable without back-propagating
// through the interior-point iterations themselves -- exactly what
// ForwardDiff.jacobian(kkt_R, ...) does in the original, just seeded on a
// 6-dof relative-pose twist here instead of a 14-dof [r;q;r;q] world state.
// -------------------------------------------------------------------------
template <typename T, typename Shape1, typename Shape2, int NX, int NS>
TVec<NX + NS, T> kktR(const Shape1& shape1, const Shape2& shape2, const TVec<NX, double>& x,
                       const TVec<NS, double>& z, const TMat<4, 4, T>& g2) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = castProblemMats<T>(problemMatrices<double>(shape1, I4));
    const auto P2 = problemMatrices<T>(shape2, g2);
    const auto combined = combineProblemMatrices<T>(P1, P2);

    static_assert(decltype(combined)::nx == NX, "kktR: nx mismatch");
    static_assert(decltype(combined)::ns == NS, "kktR: ns mismatch");
    constexpr int n_ort = decltype(combined)::n_ort;
    constexpr int n_soc1 = decltype(combined)::n_soc1;
    constexpr int n_soc2 = decltype(combined)::n_soc2;

    const TVec<NX, T> xT = x.template cast<T>();
    const TVec<NS, T> zT = z.template cast<T>();

    TVec<NX + NS, T> R;
    R.template head<NX>() = combined.c + combined.G.transpose() * zT;
    R.template tail<NS>() = cone_product<n_ort, n_soc1, n_soc2, T>(combined.h - combined.G * xT, zT);
    return R;
}

// dR/dxi (the (nx+ns) x 6 Jacobian of kktR w.r.t. shape 2's local twist,
// evaluated at xi = 0), via se3::retract<Dual6>.
template <typename Shape1, typename Shape2, int NX, int NS>
Eigen::Matrix<double, NX + NS, 6> kktRJacobian(const Shape1& shape1, const Shape2& shape2,
                                                const TVec<NX, double>& x, const TVec<NS, double>& z,
                                                const Eigen::Matrix4d& g0) {
    const TMat<4, 4, Dual6> g2D = se3::retract<Dual6>(g0, se3::seedTwist());
    const TVec<NX + NS, Dual6> RD = kktR<Dual6>(shape1, shape2, x, z, g2D);
    Eigen::Matrix<double, NX + NS, 6> J;
    for (int i = 0; i < NX + NS; ++i) J.row(i) = RD(i).grad().transpose();
    return J;
}

// diff_socp: ∂[p;alpha]/∂xi, a 4x6 Jacobian, via the closed-form implicit
// function theorem solve at the converged (x,s,z) -- ordinary double linear
// algebra, no autodiff needed here (only kktRJacobian above uses Dual6).
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 4, 6> diffSocp(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                      const StackVec<n_ort, n_soc1, n_soc2>& s,
                                      const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    constexpr int ns = n_ort + n_soc1 + n_soc2;

    const Eigen::Matrix<double, nx + ns, 6> dR_dxi = kktRJacobian<Shape1, Shape2, nx, ns>(shape1, shape2, x, z, g0);
    const Mat<nx, 6> r1 = -dR_dxi.template topRows<nx>();
    const Mat<ns, 6> r2 = -dR_dxi.template bottomRows<ns>();

    const auto P1 = problemMatrices<double>(shape1, Eigen::Matrix4d::Identity());
    const auto P2 = problemMatrices<double>(shape2, g0);
    const auto combined = combineProblemMatrices<double>(P1, P2);
    const Mat<ns, nx>& G = combined.G;

    const PlainScaling<n_ort, n_soc1, n_soc2> Z = plainScalingFromZ<n_ort, n_soc1, n_soc2>(z);
    const NTScaling<n_ort, n_soc1, n_soc2> S = scalingFromS<n_ort, n_soc1, n_soc2>(s);
    const PlainScaling<n_ort, n_soc1, n_soc2> SZ = solve(S, Z); // S \ Z

    const Mat<ns, nx> SZG = SZ.template applyMat<nx>(G);
    const Mat<nx, nx> A = G.transpose() * SZG; // G' * ((S\Z) * G) -- not symmetric in general
    const Mat<ns, 6> Sr2 = S.template solveMat<6>(r2);
    const Mat<nx, 6> rhs = r1 - G.transpose() * Sr2;

    // Julia's `\` on a general (non-symmetric) square matrix is an LU solve;
    // matched here rather than assuming/forcing symmetry with a Cholesky.
    const Eigen::Matrix<double, nx, 6> dx = A.partialPivLu().solve(rhs);
    return dx.template topRows<4>(); // drop any shape-specific extra decision vars
}

struct ProximityJacobianResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero(); // rows [wx,wy,wz,alpha], cols [v;w]
    int iters = 0;
    bool converged = false;
};

template <typename Shape1, typename Shape2>
ProximityJacobianResult proximityJacobian(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g,
                                           const SocpOptions& opt = SocpOptions{}) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto P1 = problemMatrices<double>(shape1, I4);
    const auto P2 = problemMatrices<double>(shape2, g);
    const auto combined = combineProblemMatrices<double>(P1, P2);

    const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        combined.c, combined.G, combined.h, opt);

    ProximityJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        res.jacobian = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, sol.x, sol.s, sol.z, g);
    }
    return res;
}

} // namespace dcolpp::socp
