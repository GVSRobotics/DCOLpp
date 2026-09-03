#pragma once
// dcolpp::socp::runtime_poly -- analytic derivatives of a PolytopeX-pair proximity
// solution w.r.t. shape 2's local twist xi. Runtime-row-count copies of the
// PolytopeX-relevant pieces of analytic_derivatives.hpp; the block-elimination
// IFT solve is identical (A = G'(S\Z)G stays 4x4). ORT-only: S\Z is the
// elementwise z_i/s_i, no arrow matrices.
//
// M2: PolytopeX vs PolytopeX (n_soc = 0, nx = 4). Shape 1 sits at g = Identity
// and contributes nothing; only shape 2's contribution is built.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp" // d2InvPoint / d2InvRot (inline, reused)
#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::d2InvPoint;
using dcolpp::socp::d2InvRot;

template <int MaxRows>
using MatX6 = Eigen::Matrix<double, Eigen::Dynamic, 6, 0, MaxRows, 6>;

// One PolytopeX's d(G_ort x_local)/dxi, d(h_ort)/dxi, d(G_ort^T z_ort)/dxi.
// x_local = [p; alpha] frozen; z_ort frozen. Mirrors polytopeXiDerivative.
struct ShapeXiDerivativeX {
    MatX6<PolytopeX::MaxNH> dGortX; // nh x 6
    MatX6<PolytopeX::MaxNH> dHort;  // nh x 6
    Eigen::Matrix<double, 4, 6> dGortTz = Eigen::Matrix<double, 4, 6>::Zero();
};

inline ShapeXiDerivativeX polytopeXiDerivative(const PolytopeX& shape, const Eigen::Matrix4d& g0,
                                               const Eigen::Vector3d& p, const StackVecX& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3);
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());

    ShapeXiDerivativeX out;
    // G_ort x_local = A R^T p - b alpha: p frozen, no product rule.
    out.dGortX.noalias() = shape.A * se3::dInverseRotatedVectorDXi(g0, p);
    // h_ort = A R^T r: r varies with xi -- product rule.
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHort.noalias() = shape.A * dRtr_dxi;
    // (G_ort^T z_ort) = R A^T z_ort, z_ort frozen.
    out.dGortTz.topRows<3>() = se3::dRotatedVectorDXi(g0, Eigen::Vector3d(shape.A.transpose() * z_ort));
    return out;
}

// polytopeHessianFrozen: 6x6 d/dxi[grad] with x, z frozen.
inline Eigen::Matrix<double, 6, 6> polytopeHessianFrozen(const PolytopeX& shape, const Eigen::Matrix4d& g0,
                                                         const Eigen::Vector3d& p, const StackVecX& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d p0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d a_poly = shape.A.transpose() * z_ort;
    return -(d2InvPoint(R0, p0, a_poly) - d2InvRot(R0, a_poly, p));
}

// combineXiJacobian for a hull pair (N_SOC1 = N_SOC2 = 0, V1 = V2 = 4).
struct CombinedXiJacobianX {
    static constexpr int nx = 4;
    Eigen::Matrix<double, 4, 6> dR1_dxi = Eigen::Matrix<double, 4, 6>::Zero();
    MatX6<kMaxOrtRows> dR2_dxi; // ns x 6
    MatX6<kMaxOrtRows> q;       // ns x 6
};

inline CombinedXiJacobianX combineXiJacobian(const ShapeXiDerivativeX& s2, int n_ort1, const StackVecX& z) {
    const int n_ort2 = static_cast<int>(s2.dGortX.rows());
    const int ns = n_ort1 + n_ort2;

    CombinedXiJacobianX out;
    out.q.setZero(ns, 6);
    out.q.middleRows(n_ort1, n_ort2) = s2.dHort - s2.dGortX; // q = d(h - Gx)/dxi, shape 2 rows only

    out.dR1_dxi = s2.dGortTz; // V2 = 4, no extra rows

    // dR2/dxi = Arrow(z) q, ORT -> diag(z): block-diagonal, shape-2 rows only
    out.dR2_dxi.noalias() = z.asDiagonal() * out.q;
    return out;
}

struct SensitivityResultX {
    Eigen::Matrix<double, 4, 6> dx;
    MatX6<kMaxOrtRows> ds;
    MatX6<kMaxOrtRows> dz;
};

// IFT block elimination, ORT-only: S\Z is elementwise z./s.
//   A = G' diag(z./s) G,  dx = A \ (r1 - G' (r2./s)),
//   dz = diag(z./s) G dx + r2./s,  ds = q - G dx.
inline SensitivityResultX computeSocpSensitivityFromXiJac(const CombinedXiJacobianX& xi_jac, const StackVecX& s,
                                                          const StackVecX& z, const ConstraintMatX<4>& G) {
    const Eigen::Matrix<double, 4, 6> r1 = -xi_jac.dR1_dxi;
    const MatX6<kMaxOrtRows> r2 = -xi_jac.dR2_dxi;

    const StackVecX sz = z.cwiseQuotient(s);          // S \ Z  (diag)
    const MatX6<kMaxOrtRows> Sr2 = r2.array().colwise() / s.array(); // S \ r2

    const auto SZG = (sz.asDiagonal() * G).eval();    // ns x 4
    const Eigen::Matrix4d A = G.transpose() * SZG;
    const Eigen::Matrix<double, 4, 6> rhs = r1 - G.transpose() * Sr2;

    SensitivityResultX out;
    out.dx = A.partialPivLu().solve(rhs);
    out.dz.noalias() = SZG * out.dx + Sr2;
    out.ds.noalias() = xi_jac.q - G * out.dx;
    return out;
}

// --- shape-2-slice helpers + public results ---------------------------------

// Shape 2's own z_ort slice out of the combined dual z.
inline StackVecX shape2ZOrt(const StackVecX& z, int n_ort1, int n_ort2) { return z.segment(n_ort1, n_ort2); }

// d(alpha)/dxi = -q^T z  (envelope theorem).
inline Eigen::Matrix<double, 1, 6> computeProximityGradient(const PolytopeX& shape2, const Eigen::Matrix4d& g0,
                                                            const Eigen::Vector4d& x, const StackVecX& z, int n_ort1) {
    const int n_ort2 = shape2.nh();
    const ShapeXiDerivativeX d2 = polytopeXiDerivative(shape2, g0, x.head<3>(), shape2ZOrt(z, n_ort1, n_ort2));
    const CombinedXiJacobianX xi_jac = combineXiJacobian(d2, n_ort1, z);
    return -(xi_jac.q.transpose() * z);
}

// d^2(alpha)/dxi^2 = H_frozen - r1^T dx - q^T dz.
inline Eigen::Matrix<double, 6, 6> computeProximityHessian(const PolytopeX& shape2, const Eigen::Matrix4d& g0,
                                                           const Eigen::Vector4d& x, const StackVecX& s,
                                                           const StackVecX& z, const ConstraintMatX<4>& G,
                                                           int n_ort1) {
    const int n_ort2 = shape2.nh();
    const StackVecX z_ort2 = shape2ZOrt(z, n_ort1, n_ort2);
    const ShapeXiDerivativeX d2 = polytopeXiDerivative(shape2, g0, x.head<3>(), z_ort2);
    const CombinedXiJacobianX xi_jac = combineXiJacobian(d2, n_ort1, z);
    const SensitivityResultX sens = computeSocpSensitivityFromXiJac(xi_jac, s, z, G);

    const Eigen::Matrix<double, 4, 6> r1 = -xi_jac.dR1_dxi;
    const Eigen::Matrix<double, 6, 6> H_frozen = polytopeHessianFrozen(shape2, g0, x.head<3>(), z_ort2);
    return H_frozen - r1.transpose() * sens.dx - xi_jac.q.transpose() * sens.dz;
}

// d[witness; alpha]/dxi (4x6): the first-order IFT solve.
inline Eigen::Matrix<double, 4, 6> diffSocp(const PolytopeX& shape2, const Eigen::Matrix4d& g0, const Eigen::Vector4d& x,
                                            const StackVecX& s, const StackVecX& z, const ConstraintMatX<4>& G,
                                            int n_ort1) {
    const int n_ort2 = shape2.nh();
    const ShapeXiDerivativeX d2 = polytopeXiDerivative(shape2, g0, x.head<3>(), shape2ZOrt(z, n_ort1, n_ort2));
    const CombinedXiJacobianX xi_jac = combineXiJacobian(d2, n_ort1, z);
    return computeSocpSensitivityFromXiJac(xi_jac, s, z, G).dx;
}

// One bundle: jacobian, grad, normal_jacobian -- shared intermediates built
// once (mirrors computeContactJacobianBundle).
struct ContactJacobianBundleX {
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
};

inline ContactJacobianBundleX computeContactJacobianBundle(const PolytopeX& shape2, const Eigen::Matrix4d& g0,
                                                           const Eigen::Vector4d& x, const StackVecX& s,
                                                           const StackVecX& z, const ConstraintMatX<4>& G,
                                                           int n_ort1) {
    const int n_ort2 = shape2.nh();
    const StackVecX z_ort2 = shape2ZOrt(z, n_ort1, n_ort2);
    const ShapeXiDerivativeX d2 = polytopeXiDerivative(shape2, g0, x.head<3>(), z_ort2);
    const CombinedXiJacobianX xi_jac = combineXiJacobian(d2, n_ort1, z);
    const SensitivityResultX sens = computeSocpSensitivityFromXiJac(xi_jac, s, z, G);

    ContactJacobianBundleX out;
    out.jacobian = sens.dx;
    out.grad = -(xi_jac.q.transpose() * z);

    const Eigen::Matrix<double, 4, 6> r1 = -xi_jac.dR1_dxi;
    const Eigen::Matrix<double, 6, 6> H_frozen = polytopeHessianFrozen(shape2, g0, x.head<3>(), z_ort2);
    const Eigen::Matrix<double, 6, 6> H = H_frozen - r1.transpose() * sens.dx - xi_jac.q.transpose() * sens.dz;

    const Eigen::Vector3d grad_v = out.grad.tail<3>().transpose();
    const Eigen::Matrix<double, 3, 6> dgradv_dxi = H.bottomRows<3>();
    const Eigen::Matrix3d Rg = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d u = Rg * grad_v;
    const double unorm = u.norm();
    const Eigen::Matrix<double, 3, 6> du_dxi = se3::dRotatedVectorDXi(g0, grad_v) + Rg * dgradv_dxi;
    const Eigen::Vector3d nrm = u / unorm;
    const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - nrm * nrm.transpose();
    out.normal_jacobian = (proj / unorm) * du_dxi;
    return out;
}

} // namespace dcolpp::socp::runtime_poly
