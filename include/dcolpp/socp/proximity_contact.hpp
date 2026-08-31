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
// (computeProximityGradient, O(1) after the solve -- no IFT elimination).
// Use this when Jacobians aren't needed; it's strictly cheaper than
// proximityContactJacobian.
//
// proximityContactJacobian: one SOCP solve + one computeContactJacobianBundle
// call that produces all three Jacobians (d[witness;alpha]/dxi, d(alpha)/dxi,
// d(normal)/dxi) while building shape 2's xi-derivative, the combined xi_jac,
// and the first-order IFT solve exactly once each. This is the expensive path
// -- it needs the Hessian's hessianFrozenFull internally for d(normal)/dxi --
// but allocation-free (fixed-size Eigen throughout). Optionally
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

    // Per-point jacobian/normal_jacobian, one entry per contact_manifold_
    // points[i] (same size, same order) -- populated whenever
    // opt.compute_contact_manifold is true (not a separate opt-in: see its
    // comment in solver.hpp for why these are needed at all once
    // contact_manifold_dim > 0, and for the "same active set" assumption
    // they're computed under). Empty otherwise.
    std::vector<ManifoldPointJacobian> contact_manifold_point_jacobians;
};

// (shape1, shape2, g) -> witness point, alpha, normal, and all three
// Jacobians w.r.t. xi=[w;v] (jacobian's rows are [wx,wy,wz,alpha];
// normal_jacobian's rows are the normal's [nx,ny,nz]). Strictly more work
// than proximityContact -- pulls in the full Hessian machinery for
// normal_jacobian.
//
// The optional `warm` handle warm-starts the SOCP for temporally-continuous
// queries and shares one per-pair state with proximityJacobian (see
// warm_start.hpp); passing nullptr is byte-for-byte the cold path.
//
// Same stopping rule as cold (mu < pdip_tol AND KKT residuals <= pdip_tol,
// checked on the accepted iterate) -- no early-exit trick. The seed is
// warmStartInitCentral: the previous converged (s*, z*) carried forward
// with only the exact update for the new G and a uniform lambda*e lift, so
// s o z stays proportional to e and the frozen Hessian behind
// normal_jacobian gets a genuinely central-path point. A warm solve that
// converges with mu under pdip_tol while x is still settling (residuals
// lagging) is rejected and redone cold, so normal_jacobian is never lower
// quality than the cold path.
//
// Speed: a static / slowly-drifting contact (grasp, rest, sustained
// contact -- what warm-start is for) converges in ~1 iteration. A contact
// whose pose changes appreciably per call gets the residual gate and falls
// back to a cold solve, i.e. ~cold cost, no regression.
template <typename Shape1, typename Shape2>
ProximityContactJacobianResult proximityContactJacobian(const Shape1& shape1, const Shape2& shape2,
                                                          const Eigen::Matrix4d& g,
                                                          const SocpOptions& opt = SocpOptions{},
                                                          ContactWarmState<Shape1, Shape2>* warm = nullptr) {
    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    // Body 1's problem matrices are pose-independent; cache on the warm
    // handle when present (the cold path is unchanged).
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
            // Tight gate (not proximityJacobian's loose kResidTolMul): the
            // frozen Hessian needs a genuinely cold-quality point, so require
            // the KKT residuals down to pdip_tol too -- a warm solve whose mu
            // dipped under pdip_tol while x was still settling is rejected and
            // redone cold.
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

        // Single bundle call: shape 2's xi-derivative, the combined xi_jac,
        // and the first-order IFT solve are each computed once and shared
        // across jacobian/grad/normal_jacobian. Only the Hessian's
        // hessianFrozenFull (6 directional second derivatives) is unique to
        // normal_jacobian -- see computeContactJacobianBundle.
        const auto bundle =
            computeContactJacobianBundle<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                shape1, shape2, sol.x, sol.s, sol.z, g, combined.G);
        res.jacobian = bundle.jacobian;
        res.normal = (g.block<3, 3>(0, 0) * bundle.grad.template tail<3>().transpose()).normalized();
        res.normal_jacobian = bundle.normal_jacobian;

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
                    // dim==0: the manifold IS x*, so the already-computed
                    // res.jacobian/res.normal_jacobian already apply here --
                    // no extra IFT solve needed, just mirror them.
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

            // Per-point jacobian, one entry per contact_manifold_points[i] --
            // always computed once the manifold itself is (not a separate
            // opt-in; see solver.hpp's comment by compute_contact_manifold).
            res.contact_manifold_point_jacobians.reserve(cm.witness_points.size());
            if (cm.contact_manifold_dim == 0) {
                // Single point == x* (contact_manifold.hpp's own guarantee
                // for dim 0) -- no extra IFT solve needed, mirror res.jacobian.
                res.contact_manifold_point_jacobians.push_back({res.jacobian, res.normal_jacobian});
            } else {
                // "Same active set" convention: s*/z* from the original solve
                // reused UNCHANGED at every point -- only the position (x's
                // head<3>()) is swapped in per point (see solver.hpp for why
                // recomputing s per point is wrong here). normal_jacobian is
                // NOT recomputed per point -- it's provably point-invariant
                // under this convention (verified exactly, including on
                // curved manifolds), so every entry just gets the single
                // already-computed res.normal_jacobian.
                //
                // The witness/alpha jacobian (diffSocp) DOES genuinely vary
                // per point -- but, under this same convention, it is an
                // EXACT AFFINE function of position within the patch (q, and
                // hence grad=-q^Tz and the whole IFT sensitivity RHS, depend
                // on x only through terms linear in x -- verified to machine
                // precision, all 4x6 entries, not just the alpha row). For
                // dim 2 with >=3 points that means only 3 full IFT solves are
                // needed to span the patch; every further point is a cheap
                // affine evaluation instead of its own solve. dim 1 always
                // returns exactly 2 points (the line's endpoints) -- already
                // the minimum needed to define its own affine map, so no
                // fewer than 2 full solves are possible there; only dim 2
                // with K>3 (raising contact_manifold_points for finer
                // resolution) sees the saving grow.
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
                    // Guards against the (geometrically unlikely, given
                    // contactManifold's farthest-point sampling) case where
                    // the first 3 points happen to be near-collinear -- falls
                    // back to a full solve per point rather than risk a
                    // near-singular 2x2 solve.
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
