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

// Contact normal at the witness point, in the pair's reference (shape-1)
// frame: n proportional to (d(alpha)/dp)^T -- the same envelope-theorem
// derivative `grad` carries (Le Cleac'h et al., "Single-Level Differentiable
// Contact Simulation", RAL 2023, Eq. 14), here w.r.t. the 6-dof
// relative-pose twist.
//
// `grad`'s translational block (tail<3>(), xi=[w;v]) is d(alpha)/dv with v
// a LOCAL (body-2-frame) direction, so d/d(p_ref) = g.linear() * d/dv; `g`
// must be the pose passed to alphaGradient that produced `r`.
//
// Zero vector if `r` didn't converge or that block vanishes (degenerate) --
// check `r.converged` first.
inline Eigen::Vector3d contactNormal(const AlphaGradientResult& r, const Eigen::Matrix4d& g) {
    return (g.block<3, 3>(0, 0) * r.grad.template tail<3>().transpose()).normalized();
}

// d(contact normal)/dxi (3x6): wrapper around computeContactNormalJacobian
// that deduces v1/v2/n_ort1/n_ort2 from problemMatrices (like diffSocp).
// Pulls in the Hessian machinery -- noticeably more work than contactNormal
// / alphaGradient, so only call it when d(normal)/dxi is needed.
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
    // opt.compute_degeneracy_info is set (not free, off by default). When
    // not computed the dims stay at -1 and the bools default to true, where
    // "true" means "not checked", not "confirmed valid" -- test the dims
    // against -1 to tell which. alpha's row of `jacobian` is always valid;
    // it's the witness-point rows that go bad when witness_jacobian_valid
    // is false.
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;

    // Multi-point ContactManifold -- filled only when
    // opt.compute_contact_manifold is set (implies the degeneracy info).
    // Empty otherwise; 1/2/opt.contact_manifold_points entries when set,
    // matching contact_manifold_dim 0/1/2.
    std::vector<Eigen::Vector3d> contact_manifold_points;

    // Per-point jacobian/normal_jacobian, aligned with
    // contact_manifold_points (same size/order) -- filled whenever the
    // manifold is (see solver.hpp for why they're needed once
    // contact_manifold_dim > 0, and the "same active set" assumption they
    // use). Empty otherwise.
    std::vector<ManifoldPointJacobian> contact_manifold_point_jacobians;
};

// (shape1, shape2, g) -> witness point, alpha, normal, and their Jacobians
// w.r.t. xi=[w;v] (jacobian rows [wx,wy,wz,alpha]; normal_jacobian rows
// [nx,ny,nz]). More work than proximityContact -- normal_jacobian pulls in
// the frozen Hessian.
//
// The optional `warm` handle warm-starts the SOCP for temporally-continuous
// queries, sharing per-pair state with proximityJacobian (warm_start.hpp);
// nullptr is the cold path. It seeds from warmStartInitCentral (previous
// (s*, z*) carried forward with a uniform lift, so s o z stays proportional
// to e -- normal_jacobian's frozen Hessian needs a central-path point) and
// keeps the cold stopping rule (mu AND KKT residuals <= pdip_tol on the
// accepted iterate); a warm solve whose residuals lag is redone cold. A
// static / slowly-drifting contact converges in ~1 iteration; a fast-moving
// one falls back to a cold solve.
template <typename Shape1, typename Shape2>
ProximityContactJacobianResult proximityContactJacobian(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g,
                                                          const SocpOptions& opt = SocpOptions{},
                                                          ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    // Body 1's problem matrices are pose-independent; cache them on the warm
    // handle when present, else rebuild each call.
    using P1T = decltype(problemMatrices(shape1, I4));
    P1T P1_local;
    const P1T* P1p = &P1_local;
    if (warm != nullptr) {
        if (!warm->P1_valid) {
            warm->P1 = problemMatrices(shape1, I4);
            warm->P1_valid = true;
        }
        P1p = &warm->P1;
    } else {
        P1_local = problemMatrices(shape1, I4);
    }
    const auto& P1 = *P1p;

    const auto P2 = problemMatrices(shape2, g);
    const auto combined = combineProblemMatrices(P1, P2);

    SocpResult<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx> sol;
    bool warm_used = false;
    const double pose_move = (warm != nullptr && warm->valid) ? poseMoveMetric(warm->g_ref, g) : 0.0;
    if (warm != nullptr && warm->valid && pose_move <= warm->rho) {
        const auto wi = warmStartInitCentral<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, warm->x, warm->z);
        SocpOptions wopt = opt;
        wopt.max_iters = std::min(opt.max_iters, WarmStartConfig::kMaxIters);
        sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, wopt, &wi);
        if (sol.converged) {
            // normal_jacobian's frozen Hessian needs a cold-quality point,
            // so require the KKT residuals down to pdip_tol too (tighter
            // than proximityJacobian's gate); a warm solve that misses this
            // is redone cold.
            const double rx = (combined.G.transpose() * sol.z + combined.c).norm();
            const double rz = (sol.s + combined.G * sol.x - combined.h).norm();
            warm_used = (rx <= opt.pdip_tol && rz <= opt.pdip_tol);
        }
    }
    if (!warm_used) {
        sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            shape1, shape2, g, combined.c, combined.G, combined.h, opt);
        if (warm != nullptr && warm->valid)
            warm->rho = std::max(WarmStartConfig::kRhoShrink * warm->rho, WarmStartConfig::kRhoMin);
    } else if (warm != nullptr && sol.iters <= WarmStartConfig::kCheapIters) {
        warm->rho = std::min(WarmStartConfig::kRhoGrow * warm->rho, WarmStartConfig::kRhoMax);
    }

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

    if (warm != nullptr) {
        if (res.converged) {
            warm->g_ref = g;
            warm->x = sol.x;
            warm->s = sol.s;
            warm->z = sol.z;
            warm->mu = sol.mu;
            warm->valid = true;
        } else {
            warm->valid = false;
        }
    }
    return res;
}

} // namespace dcolpp::socp
