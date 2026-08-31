#pragma once
// dcolpp::socp -- the contact layer on top of proximity.hpp: the contact
// normal, and the two bundled queries a contact simulator wants.
//
//   contactNormal / contactNormalJacobian -- n_hat and d(n_hat)/dxi
//   proximityContact         -- witness, alpha, normal (one SOCP solve)
//   proximityContactJacobian -- + d[witness;alpha]/dxi, d(alpha)/dxi,
//                               d(normal)/dxi (the last needs the frozen
//                               Hessian); optionally the ContactDegeneracy
//                               diagnostics and/or the ContactManifold
//                               multi-point witness set for a degenerate
//                               (line / face) contact
//
// A thin composition of proximity.hpp / analytic_derivatives.hpp /
// contact_degeneracy.hpp / contact_manifold.hpp; no new derivative math here.

#include <Eigen/Dense>
#include <vector>

#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/contact_degeneracy.hpp"
#include "dcolpp/socp/contact_manifold.hpp"
#include "dcolpp/socp/proximity.hpp"

namespace dcolpp::socp {

// Contact normal at witness point in shape-1 frame (proportional to d(alpha)/dp^T).
// `grad`'s translational block is d(alpha)/dv (LOCAL body-2-frame), transformed
// via g.linear(). Returns zero vector if unconverged or degenerate -- check r.converged.
inline Eigen::Vector3d contactNormal(const AlphaGradientResult& r, const Eigen::Matrix4d& g) {
    return (g.block<3, 3>(0, 0) * r.grad.template tail<3>().transpose()).normalized();
}

// d(contact normal)/dxi (3x6): wrapper around computeContactNormalJacobian.
// Deduces v1/v2/n_ort1/n_ort2 from problemMatrices. More expensive than contactNormal.
template <typename Shape1, typename Shape2, int n_ort, int n_soc1, int n_soc2, int nx>
Eigen::Matrix<double, 3, 6> contactNormalJacobian(const Shape1& shape1, const Shape2& shape2, const DecisionVec<nx>& x,
                                                   const StackVec<n_ort, n_soc1, n_soc2>& s,
                                                   const StackVec<n_ort, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
    constexpr int v2 = nx - v1 + 4;
    constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
    constexpr int n_ort2 = n_ort - n_ort1;
    return computeContactNormalJacobian<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape1, shape2, x, s, z,
                                                                                                   g0);
}

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
    const auto gr = alphaGradient(shape1, shape2, g, opt);

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

// Jacobian pair at one contact_manifold_points[i], computed under the
// "assume same active set as x*" convention -- see solver.hpp's
// compute_contact_manifold comment.
struct ManifoldPointJacobian {
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();        // d[point;alpha]/dxi at this point
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero(); // d(normal)/dxi at this point
};

struct ProximityContactJacobianResult {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();        // d[witness;alpha]/dxi
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero(); // d(normal)/dxi
    int iters = 0;
    bool converged = false;

    // Degeneracy diagnostics (ContactDegeneracy) -- filled only when
    // opt.compute_degeneracy_info is set. Dims stay at -1 if not computed.
    // alpha's row of `jacobian` is always valid; witness-point rows may be
    // invalid if witness_jacobian_valid is false.
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;

    // Multi-point ContactManifold -- filled when opt.compute_contact_manifold is set.
    std::vector<Eigen::Vector3d> contact_manifold_points;

    // Per-point jacobian/normal_jacobian, aligned with contact_manifold_points.
    // Empty if contact_manifold_dim <= 0.
    std::vector<ManifoldPointJacobian> contact_manifold_point_jacobians;
};

// Computes witness point, alpha, normal, and their Jacobians w.r.t. xi=[w;v].
// Jacobian rows: [wx,wy,wz,alpha]; normal_jacobian rows: [nx,ny,nz].
// Optional `warm` handle enables warm-starts for temporally-continuous queries;
// nullptr uses cold solve. Warm-starts seed from central path and converge in
// ~1 iteration for static contacts; fast-moving contacts fall back to cold solve.
template <typename Shape1, typename Shape2>
ProximityContactJacobianResult proximityContactJacobian(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g,
                                                          const SocpOptions& opt = SocpOptions{},
                                                          ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    decltype(problemMatrices(shape1, Eigen::Matrix4d::Identity())) P1_local;
    const auto& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    // Central seed and the tight residual gate (resid_tol_mul = 1.0):
    // normal_jacobian's frozen Hessian needs a genuinely central-path point,
    // not just the interior one proximityJacobian's gate accepts.
    const auto sol = solveForQuery<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        shape1, shape2, g, combined.c, combined.G, combined.h, opt, warm, WarmSeed::Central, 1.0);

    ProximityContactJacobianResult res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.template head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

        // One bundle call: shape 2's xi-derivative, the combined xi_jac, and
        // the first-order IFT solve are each done once and shared across
        // jacobian/grad/normal_jacobian; only hessianFrozenFull (one 6x6) is
        // unique to normal_jacobian.
        const auto bundle =
            computeContactJacobianBundle<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);
        res.jacobian = bundle.jacobian;
        res.normal = (g.block<3, 3>(0, 0) * bundle.grad.template tail<3>().transpose()).normalized();
        res.normal_jacobian = bundle.normal_jacobian;

        if constexpr (IsStrictlyConvex<Shape1>::value || IsStrictlyConvex<Shape2>::value) {
            // If either shape is strictly convex (Sphere/Ellipsoid, see
            // primitives.hpp: IsStrictlyConvex), the contact is provably
            // contact_manifold_dim==0 / normal_cone_dim==0 -- skip the
            // SVD/LU machinery entirely when a diagnostic was requested.
            if (opt.compute_contact_manifold || opt.compute_degeneracy_info) {
                res.contact_manifold_dim = 0;
                res.normal_cone_dim = 0;
                res.witness_jacobian_valid = true;
                res.normal_jacobian_valid = true;
                if (opt.compute_contact_manifold) {
                    res.contact_manifold_points.push_back(res.witness_point);
                    // dim==0: the manifold IS x*, so mirror the jacobians
                    // already computed -- no extra IFT solve.
                    res.contact_manifold_point_jacobians.push_back({res.jacobian, res.normal_jacobian});
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

            // Per-point jacobians, computed whenever the manifold is (see
            // solver.hpp by compute_contact_manifold).
            res.contact_manifold_point_jacobians.reserve(cm.witness_points.size());
            if (cm.contact_manifold_dim == 0) {
                // Single point == x* -- mirror res.jacobian, no extra solve.
                res.contact_manifold_point_jacobians.push_back({res.jacobian, res.normal_jacobian});
            } else {
                // "Same active set": s*/z* reused, only position swapped per point.
                // normal_jacobian is point-invariant. For dim 2 with >3 points,
                // affine combinations of 3 IFT solves span the manifold.
                auto fullAt = [&](const Eigen::Vector3d& p) {
                    DecisionVec<combined.nx> xp = sol.x;
                    xp.template head<3>() = p;
                    ManifoldPointJacobian mpj;
                    mpj.jacobian =
                        diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
                            shape1, shape2, xp, sol.s, sol.z, g, combined.G);
                    mpj.normal_jacobian = res.normal_jacobian;
                    return mpj;
                };

                if (cm.contact_manifold_dim == 2 && cm.witness_points.size() > 3) {
                    const Eigen::Vector3d p0 = cm.witness_points[0];
                    const Eigen::Vector3d e1 = cm.witness_points[1] - p0;
                    const Eigen::Vector3d e2 = cm.witness_points[2] - p0;
                    Eigen::Matrix<double, 3, 2> E;
                    E.col(0) = e1;
                    E.col(1) = e2;
                    const Eigen::Matrix2d EtE = E.transpose() * E;
                    const double detEtE = EtE.determinant();
                    // If the first 3 points are near-collinear, fall back to
                    // a full solve per point rather than a near-singular 2x2.
                    const bool spanOk =
                        std::abs(detEtE) > 1e-10 * std::max(e1.squaredNorm() * e2.squaredNorm(), 1e-24);

                    const ManifoldPointJacobian m0 = fullAt(p0);
                    const ManifoldPointJacobian m1 = fullAt(cm.witness_points[1]);
                    const ManifoldPointJacobian m2 = fullAt(cm.witness_points[2]);
                    res.contact_manifold_point_jacobians.push_back(m0);
                    res.contact_manifold_point_jacobians.push_back(m1);
                    res.contact_manifold_point_jacobians.push_back(m2);

                    if (spanOk) {
                        const Eigen::Matrix2d EtE_inv = EtE.inverse();
                        const Eigen::Matrix<double, 4, 6> D1 = m1.jacobian - m0.jacobian;
                        const Eigen::Matrix<double, 4, 6> D2 = m2.jacobian - m0.jacobian;
                        for (size_t i = 3; i < cm.witness_points.size(); ++i) {
                            const Eigen::Vector3d d = cm.witness_points[i] - p0;
                            const Eigen::Vector2d coeffs = EtE_inv * (E.transpose() * d);
                            ManifoldPointJacobian mpj;
                            mpj.jacobian = m0.jacobian + coeffs(0) * D1 + coeffs(1) * D2;
                            mpj.normal_jacobian = res.normal_jacobian;
                            res.contact_manifold_point_jacobians.push_back(mpj);
                        }
                    } else {
                        for (size_t i = 3; i < cm.witness_points.size(); ++i) {
                            res.contact_manifold_point_jacobians.push_back(fullAt(cm.witness_points[i]));
                        }
                    }
                } else {
                    for (const Eigen::Vector3d& p : cm.witness_points) {
                        res.contact_manifold_point_jacobians.push_back(fullAt(p));
                    }
                }
            }
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
