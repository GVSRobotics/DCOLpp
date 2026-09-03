#pragma once
// dcolpp::socp::runtime_poly -- contact layer for a PolytopeX pair: witness point,
// alpha, contact normal, per-body scale-back witnesses + gap, and their
// xi-Jacobians. Mirrors contact.hpp for the hull-hull case.
//
// Opt-in contact manifold / degeneracy for a face-face (patch) contact:
// SocpOptions::compute_contact_manifold / compute_degeneracy_info. Mirrors
// contact.hpp's detectManifold; the multi-point set + per-point Jacobians make
// a flush hull-hull or hull-on-plane contact resolvable without a jittering
// single witness. Pure-ORT pairs only (runtime_poly/contact_manifold.hpp).

#include <Eigen/Dense>
#include <vector>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp" // dcolpp::socp::ContactWitnesses / contactWitnesses (dim-free, reused)
#include "dcolpp/socp/runtime_poly/contact_manifold.hpp"
#include "dcolpp/socp/runtime_poly/proximity.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::ContactWitnesses;
using dcolpp::socp::contactWitnesses;

struct ProximityContactResultX {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d witness_body1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d witness_body2 = Eigen::Vector3d::Zero();
    double gap = 0.0;
    int iters = 0;
    bool converged = false;
    bool plane_flipped = false;

    // Filled only when opt.compute_degeneracy_info / compute_contact_manifold
    // is set (dims stay -1 otherwise). *_valid == false means the witness
    // point / normal is non-unique.
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
    std::vector<Eigen::Vector3d> contact_manifold_points;
    std::vector<ContactWitnesses> contact_manifold_witnesses; // value-only per point
};

// contact.hpp detectManifold, pure-ORT: fill degeneracy dims + (opt-in) the
// multi-point value-only manifold from the converged (x, s, z, G).
inline void fillManifoldValues(ProximityContactResultX& res, const SocpOptions& opt, const Eigen::Vector4d& x,
                               const StackVecX& s, const StackVecX& z, const ConstraintMatX<4>& G, double alpha,
                               const Eigen::Vector3d& r_org, bool plane_body1) {
    if (opt.compute_contact_manifold) {
        const ContactManifoldX cm = contactManifold(x, s, z, G, opt.contact_manifold_points);
        res.contact_manifold_dim = cm.contact_manifold_dim;
        res.normal_cone_dim = cm.normal_cone_dim;
        res.witness_jacobian_valid = cm.witness_jacobian_valid;
        res.normal_jacobian_valid = cm.normal_jacobian_valid;
        res.contact_manifold_points = cm.witness_points;
    } else if (opt.compute_degeneracy_info) {
        const ContactDegeneracyX d = contactDegeneracy(s, z, G);
        res.contact_manifold_dim = d.contact_manifold_dim;
        res.normal_cone_dim = d.normal_cone_dim;
        res.witness_jacobian_valid = d.witness_jacobian_valid;
        res.normal_jacobian_valid = d.normal_jacobian_valid;
    }
    res.contact_manifold_witnesses.reserve(res.contact_manifold_points.size());
    for (const auto& p : res.contact_manifold_points)
        res.contact_manifold_witnesses.push_back(contactWitnesses(p, alpha, r_org, plane_body1));
}

inline ProximityContactResultX proximityContact(const PolytopeX& shape1, const PolytopeX& shape2,
                                                const Eigen::Matrix4d& g, const SocpOptions& opt = SocpOptions{},
                                                ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    const CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocp(shape1.bounding_sphere, shape2.bounding_sphere, g, combined.c, combined.G,
                                        combined.h, opt); });

    ProximityContactResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        const Eigen::Vector3d r_org = g.block<3, 1>(0, 3);
        const Eigen::Matrix<double, 1, 6> grad =
            computeProximityGradient(shape2, g, sol.x, sol.z, combined.n_ort1);
        res.normal = (g.block<3, 3>(0, 0) * grad.tail<3>().transpose()).normalized();

        const ContactWitnesses w = contactWitnesses(res.witness_point, res.alpha, r_org, /*plane_body1=*/false);
        res.witness_body1 = w.body1;
        res.witness_body2 = w.body2;
        res.gap = w.gap;
        fillManifoldValues(res, opt, sol.x, sol.s, sol.z, combined.G, res.alpha, r_org, /*plane_body1=*/false);
    }
    return res;
}

// Plane (shape 1) vs PolytopeX (shape 2). plane_body1 == true in the
// scale-back witnesses; the normal carries the plane-flip sign.
inline ProximityContactResultX proximityContact(const Plane& plane, const PolytopeX& shape2, const Eigen::Matrix4d& g,
                                                const SocpOptions& opt = SocpOptions{},
                                                ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(plane, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const bool flipped = applyPlaneFlipX(plane, g, combined.G, combined.h);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocpPlane(plane.normal, plane.d, shape2.bounding_sphere, g, combined.c,
                                             combined.G, combined.h, opt); });

    ProximityContactResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    res.plane_flipped = flipped;
    if (sol.converged) {
        const Eigen::Vector3d r_org = g.block<3, 1>(0, 3);
        const Eigen::Matrix<double, 1, 6> grad =
            computeProximityGradient(shape2, g, sol.x, sol.z, combined.n_ort1);
        res.normal = (g.block<3, 3>(0, 0) * grad.tail<3>().transpose()).normalized();
        const ContactWitnesses w = contactWitnesses(res.witness_point, res.alpha, r_org, /*plane_body1=*/true);
        res.witness_body1 = w.body1;
        res.witness_body2 = w.body2;
        res.gap = w.gap;
        fillManifoldValues(res, opt, sol.x, sol.s, sol.z, combined.G, res.alpha, r_org, /*plane_body1=*/true);
    }
    return res;
}

struct ProximityContactJacobianResultX {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();        // d[witness; alpha]/dxi
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero(); // d(normal)/dxi

    Eigen::Vector3d witness_body1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d witness_body2 = Eigen::Vector3d::Zero();
    double gap = 0.0;
    Eigen::Matrix<double, 3, 6> witness_body1_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Matrix<double, 3, 6> witness_body2_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Matrix<double, 1, 6> gap_jacobian = Eigen::Matrix<double, 1, 6>::Zero();

    int iters = 0;
    bool converged = false;
    bool plane_flipped = false;

    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
    std::vector<Eigen::Vector3d> contact_manifold_points;
    std::vector<Eigen::Matrix<double, 4, 6>> contact_manifold_point_jacobians;        // d[point;alpha]/dxi per point
    std::vector<Eigen::Matrix<double, 3, 6>> contact_manifold_point_normal_jacobians; // point-invariant copy
    std::vector<ContactWitnesses> contact_manifold_witnesses;
};

// contact.hpp proximityContactJacobian's non-strict-convex manifold branch,
// pure-ORT: the multi-point set and each point's [point;alpha] Jacobian
// (normal_jacobian is point-invariant, mirrored). dim-2 patches with >3 points
// use the affine-span shortcut over 3 full IFT solves.
inline void fillManifoldJacobians(ProximityContactJacobianResultX& res, const SocpOptions& opt,
                                  const PolytopeX& shape2, const Eigen::Matrix4d& g, const Eigen::Vector4d& x,
                                  const StackVecX& s, const StackVecX& z, const ConstraintMatX<4>& G, int n_ort1,
                                  const Eigen::Matrix<double, 3, 6>& dr_dxi, double alpha,
                                  const Eigen::Vector3d& r_org, bool plane_body1) {
    if (opt.compute_contact_manifold) {
        const ContactManifoldX cm = contactManifold(x, s, z, G, opt.contact_manifold_points);
        res.contact_manifold_dim = cm.contact_manifold_dim;
        res.normal_cone_dim = cm.normal_cone_dim;
        res.witness_jacobian_valid = cm.witness_jacobian_valid;
        res.normal_jacobian_valid = cm.normal_jacobian_valid;
        res.contact_manifold_points = cm.witness_points;

        auto fullAt = [&](const Eigen::Vector3d& p) -> Eigen::Matrix<double, 4, 6> {
            Eigen::Vector4d xp = x;
            xp.head<3>() = p;
            return diffSocp(shape2, g, xp, s, z, G, n_ort1);
        };

        const auto& pts = cm.witness_points;
        if (cm.contact_manifold_dim == 0 || pts.size() <= 1) {
            for (size_t i = 0; i < pts.size(); ++i) {
                res.contact_manifold_point_jacobians.push_back(res.jacobian);
                res.contact_manifold_point_normal_jacobians.push_back(res.normal_jacobian);
            }
        } else if (cm.contact_manifold_dim == 2 && pts.size() > 3) {
            const Eigen::Vector3d p0 = pts[0];
            const Eigen::Vector3d e1 = pts[1] - p0, e2 = pts[2] - p0;
            Eigen::Matrix<double, 3, 2> E;
            E.col(0) = e1;
            E.col(1) = e2;
            const Eigen::Matrix2d EtE = E.transpose() * E;
            const double detEtE = EtE.determinant();
            const bool spanOk =
                std::abs(detEtE) > 1e-10 * std::max(e1.squaredNorm() * e2.squaredNorm(), 1e-24);
            const Eigen::Matrix<double, 4, 6> m0 = fullAt(p0), m1 = fullAt(pts[1]), m2 = fullAt(pts[2]);
            res.contact_manifold_point_jacobians = {m0, m1, m2};
            if (spanOk) {
                const Eigen::Matrix2d EtEi = EtE.inverse();
                const Eigen::Matrix<double, 4, 6> D1 = m1 - m0, D2 = m2 - m0;
                for (size_t i = 3; i < pts.size(); ++i) {
                    const Eigen::Vector2d cf = EtEi * (E.transpose() * (pts[i] - p0));
                    res.contact_manifold_point_jacobians.push_back(m0 + cf(0) * D1 + cf(1) * D2);
                }
            } else {
                for (size_t i = 3; i < pts.size(); ++i)
                    res.contact_manifold_point_jacobians.push_back(fullAt(pts[i]));
            }
            res.contact_manifold_point_normal_jacobians.assign(pts.size(), res.normal_jacobian);
        } else {
            for (const auto& p : pts) {
                res.contact_manifold_point_jacobians.push_back(fullAt(p));
                res.contact_manifold_point_normal_jacobians.push_back(res.normal_jacobian);
            }
        }
    } else if (opt.compute_degeneracy_info) {
        const ContactDegeneracyX d = contactDegeneracy(s, z, G);
        res.contact_manifold_dim = d.contact_manifold_dim;
        res.normal_cone_dim = d.normal_cone_dim;
        res.witness_jacobian_valid = d.witness_jacobian_valid;
        res.normal_jacobian_valid = d.normal_jacobian_valid;
    }

    res.contact_manifold_witnesses.reserve(res.contact_manifold_points.size());
    for (size_t i = 0; i < res.contact_manifold_points.size(); ++i)
        res.contact_manifold_witnesses.push_back(contactWitnesses(
            res.contact_manifold_points[i], alpha, r_org, res.contact_manifold_point_jacobians[i], dr_dxi, plane_body1));
}

inline ProximityContactJacobianResultX proximityContactJacobian(const PolytopeX& shape1, const PolytopeX& shape2,
                                                                const Eigen::Matrix4d& g,
                                                                const SocpOptions& opt = SocpOptions{},
                                                                ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    const CombinedProblemX combined = combineProblemMatrices(P1, P2);
    // Central seed + tight residual gate: the frozen Hessian in normal_jacobian
    // needs a genuinely central-path point.
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Central, 1.0, g,
        [&] { return solveProximitySocp(shape1.bounding_sphere, shape2.bounding_sphere, g, combined.c, combined.G,
                                        combined.h, opt); });

    ProximityContactJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged) {
        const ContactJacobianBundleX bundle =
            computeContactJacobianBundle(shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1);
        res.jacobian = bundle.jacobian;
        res.normal = (g.block<3, 3>(0, 0) * bundle.grad.tail<3>().transpose()).normalized();
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
        fillManifoldJacobians(res, opt, shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1, dr_dxi,
                              res.alpha, r_org, /*plane_body1=*/false);
    }
    return res;
}

inline ProximityContactJacobianResultX proximityContactJacobian(const Plane& plane, const PolytopeX& shape2,
                                                                const Eigen::Matrix4d& g,
                                                                const SocpOptions& opt = SocpOptions{},
                                                                ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(plane, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const bool flipped = applyPlaneFlipX(plane, g, combined.G, combined.h);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Central, 1.0, g,
        [&] { return solveProximitySocpPlane(plane.normal, plane.d, shape2.bounding_sphere, g, combined.c,
                                             combined.G, combined.h, opt); });

    ProximityContactJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    res.plane_flipped = flipped;
    if (sol.converged) {
        const ContactJacobianBundleX bundle =
            computeContactJacobianBundle(shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1);
        res.jacobian = bundle.jacobian;
        res.normal = (g.block<3, 3>(0, 0) * bundle.grad.tail<3>().transpose()).normalized();
        res.normal_jacobian = bundle.normal_jacobian;

        const Eigen::Vector3d r_org = g.block<3, 1>(0, 3);
        const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g, Eigen::Vector3d::Zero());
        const ContactWitnesses w =
            contactWitnesses(res.witness_point, res.alpha, r_org, res.jacobian, dr_dxi, /*plane_body1=*/true);
        res.witness_body1 = w.body1;
        res.witness_body2 = w.body2;
        res.gap = w.gap;
        res.witness_body1_jacobian = w.body1_jacobian;
        res.witness_body2_jacobian = w.body2_jacobian;
        res.gap_jacobian = w.gap_jacobian;
        fillManifoldJacobians(res, opt, shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1, dr_dxi,
                              res.alpha, r_org, /*plane_body1=*/true);
    }
    return res;
}

} // namespace dcolpp::socp::runtime_poly
