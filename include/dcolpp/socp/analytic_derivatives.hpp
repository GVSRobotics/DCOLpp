#pragma once
// dcolpp::socp -- analytic derivatives of the SOCP solution w.r.t. shape
// 2's local twist xi. Built from se3::dPointDXi / dRotatedVectorDXi /
// dInverseRotatedVectorDXi (generic pose-derivative primitives) plus one
// hand-derived chain-rule function per shape.
//
// Shape 1 sits at the fixed reference pose (g = Identity), so it never
// depends on xi and contributes nothing; only shape 2's contribution is
// computed.
//
// Layers, in the order they appear below:
//   1. ShapeXiDerivative + one *XiDerivative per shape: d(G*x)/dxi,
//      d(h)/dxi, d(G^T*z)/dxi, with x, z frozen at their converged values.
//   2. *HessianFrozen per shape: (1) differentiated once more w.r.t. xi,
//      still with x, z frozen (H_frozen).
//   3. combineXiJacobian / hessianFrozenFull: place a shape's (1)/(2)
//      output into the combined both-shape system.
//   4. computeSocpSensitivity(Auto) / computeProximityGradient /
//      computeProximityHessian / computeContactNormalJacobian: the
//      public results -- dx/ds/dz per dxi, d(alpha)/dxi, d^2(alpha)/dxi^2,
//      d(contact normal)/dxi.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/combine_problem_matrices.hpp"
#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/primitives.hpp"
#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

// One shape's own d(G*x_local)/dxi, d(h)/dxi, d(G^T*z_local)/dxi, split by
// ORT/SOC block (x_local, z_local are that shape's own slice of the
// combined decision/dual vectors; for a v=4 shape, x_local = [p;alpha]).
template <int N_ORT, int N_SOC, int V>
struct ShapeXiDerivative {
    Eigen::Matrix<double, N_ORT, 6> dGortX = Eigen::Matrix<double, N_ORT, 6>::Zero();
    Eigen::Matrix<double, N_ORT, 6> dHort = Eigen::Matrix<double, N_ORT, 6>::Zero();
    Eigen::Matrix<double, N_SOC, 6> dGsocX = Eigen::Matrix<double, N_SOC, 6>::Zero();
    Eigen::Matrix<double, N_SOC, 6> dHsoc = Eigen::Matrix<double, N_SOC, 6>::Zero();
    Eigen::Matrix<double, V, 6> dGortTz = Eigen::Matrix<double, V, 6>::Zero();
    Eigen::Matrix<double, V, 6> dGsocTz = Eigen::Matrix<double, V, 6>::Zero();
};

// R = Rg*R_offset is the placed rotation, r the placed origin (see
// placeShape). Each function returns only shape 2's contribution; shape 1
// is at g=Identity and contributes nothing.

// Sphere: G_soc constant (no R-dependence); only h_soc's placed-center row
// varies with xi, via se3::dPointDXi.
ShapeXiDerivative<0, 4, 4> sphereXiDerivative(const Sphere& shape, const Eigen::Matrix4d& g0);

// Capsule: axis bx = R*(1,0,0) -> se3::dRotatedVectorDXi. `t` the converged
// axial decision variable, `z_soc` the SOC dual slice. G_ort/h_ort constant.
ShapeXiDerivative<2, 4, 5> capsuleXiDerivative(const Capsule& shape, const Eigen::Matrix4d& g0, double t,
                                                const Eigen::Vector4d& z_soc);

// Cylinder: SOC block as Capsule's. G_ort adds the flat end-cap clips via
// bx; h_ort = (0, 0, -bx.r, bx.r) needs the product rule (bx and r both
// vary with xi). `p` the converged trial point, `t` the axial entry.
ShapeXiDerivative<4, 4, 5> cylinderXiDerivative(const Cylinder& shape, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                 const Eigen::Vector4d& z_ort);

// Cone: G_soc = -E*R^T -> se3::dInverseRotatedVectorDXi, plus the product
// rule wherever R^T multiplies the placed center r (xi-dependent) rather
// than a frozen decision variable. `z_ort` is the 1-vector base-cap dual.
ShapeXiDerivative<1, 3, 4> coneXiDerivative(const Cone& shape, const Eigen::Matrix4d& g0, const Eigen::Vector3d& p,
                                             double z_ort, const Eigen::Vector3d& z_soc);

// TruncatedCone: SOC block identical to Cone's; the two orthant rows are
// the +-bx flat-cap clips (Cylinder's rows 2/3). `z_ort` row 0 = base cap,
// row 1 = top cap.
ShapeXiDerivative<2, 3, 4> truncatedConeXiDerivative(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                      const Eigen::Vector3d& p, const Eigen::Vector2d& z_ort,
                                                      const Eigen::Vector3d& z_soc);

// Ellipsoid: G_soc = -U*R^T, same pattern as Cone's -E*R^T (U for E).
ShapeXiDerivative<0, 4, 4> ellipsoidXiDerivative(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                  const Eigen::Vector3d& p, const Eigen::Vector4d& z_soc);

// Polytope<NH>: G_ort = A*R^T, same pattern (A for E/U), no SOC block.
// Templated on NH to stay header-only.
template <int NH>
ShapeXiDerivative<NH, 0, 4> polytopeXiDerivative(const Polytope<NH>& shape, const Eigen::Matrix4d& g0,
                                                  const Eigen::Vector3d& p, const Eigen::Matrix<double, NH, 1>& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);

    ShapeXiDerivative<NH, 0, 4> out;
    // G_ort*x_local = A*R^T*p - b*alpha: p frozen, no product rule.
    out.dGortX = shape.A * shape.R_offset.transpose() * se3::dInverseRotatedVectorDXi(g0, p);
    // h_ort = A*R^T*r: r varies with xi -- product rule.
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHort = shape.A * shape.R_offset.transpose() * dRtr_dxi;
    // (G_ort^T*z_ort) = R*A^T*z_ort = Rg*(R_offset*A^T*z_ort), z_ort frozen.
    out.dGortTz.template block<3, 6>(0, 0) =
        se3::dRotatedVectorDXi(g0, shape.R_offset * shape.A.transpose() * z_ort);
    return out;
}

// Polygon<NH>: G_ort constant (no contribution, like Capsule). G_soc uses
// Rtilde = R's first two columns, always contracted against the frozen
// in-plane coordinate u = (u1, u2) or a frozen z -- plain dRotatedVectorDXi
// calls. Templated on NH, like polytopeXiDerivative.
template <int NH>
ShapeXiDerivative<NH, 4, 6> polygonXiDerivative(const Polygon<NH>& shape, const Eigen::Matrix4d& g0, double u1,
                                                 double u2, const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Vector3d u_local(u1, u2, 0.0);

    ShapeXiDerivative<NH, 4, 6> out;
    // G_soc*x_local's Rtilde*u term (p, u1, u2 frozen):
    //   Rtilde*u = R*[u;0] = Rg*(R_offset*[u1,u2,0]).
    out.dGsocX.template block<3, 6>(1, 0) = se3::dRotatedVectorDXi(g0, shape.R_offset * u_local);
    out.dHsoc.template block<3, 6>(1, 0) = -dr_dxi; // h_soc = (0; -r)
    // (G_soc^T*z_soc)'s u1,u2 columns = Rtilde's columns . z_vec, z_vec frozen.
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    out.dGsocTz.row(4) = z_vec.transpose() * se3::dRotatedVectorDXi(g0, shape.R_offset.col(0));
    out.dGsocTz.row(5) = z_vec.transpose() * se3::dRotatedVectorDXi(g0, shape.R_offset.col(1));
    return out;
}

// H_frozen (per shape): the 6x6 d/dxi[grad(xi)] with x, z frozen at their
// converged values, where grad = -q^T z (computeProximityGradient below).
// Built in closed form -- no e_j probing loop, no second-derivative SE(3)
// primitives -- by differentiating each shape's own -z^T q(xi) formula once
// more w.r.t. xi. Every pose-dependent factor of q^T z is a frozen covector
// a times one of four SE(3) field-Jacobian patterns; the stk* helpers are
// the closed form of d/dxi of exactly that product (column j = the
// derivative along xi_j), so each *HessianFrozen is a short linear
// combination of stk* terms. The shapes whose first derivative already
// needed a product rule (Cylinder/Cone/TruncatedCone ORT rows) add the two
// constant outer products of first-derivative Jacobians.
//
//   a_r := R0^T a  (R0 = g0's rotation, p0 = g0's translation).
//   stkPoint(a, c)    d/dxi[ a^T d(R c + p)/dxi ]   = [ c_tilde a_r_tilde | 0 ;  a_r_tilde | 0 ]
//   stkRot(a, c)      d/dxi[ a^T d(R c)/dxi ]       = [ c_tilde a_r_tilde | 0 ;  0 | 0 ]
//   stkInvRot(a, w)   d/dxi[ a^T d(R^T w)/dxi ]     = [ a_tilde (R0^T w)_tilde | 0 ;  0 | 0 ]   (w fixed)
//   stkInvPoint(a)    d/dxi[ a^T d(R^T(R c + p))/dxi ] = [ a_tilde (R0^T p0)_tilde | a_tilde ;  0 | 0 ]
//     (independent of the local point c: R^T(R c + p) = c + R^T p, c is xi-constant)
inline Eigen::Matrix<double, 6, 6> stkPoint(const Eigen::Matrix3d& R0, const Eigen::Vector3d& a,
                                            const Eigen::Vector3d& c) {
    const Eigen::Matrix3d Ar = se3::skew<double>(Eigen::Vector3d(R0.transpose() * a));
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<3, 3>(0, 0) = se3::skew<double>(c) * Ar;
    H.block<3, 3>(3, 0) = Ar;
    return H;
}
inline Eigen::Matrix<double, 6, 6> stkRot(const Eigen::Matrix3d& R0, const Eigen::Vector3d& a,
                                          const Eigen::Vector3d& c) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<3, 3>(0, 0) = se3::skew<double>(c) * se3::skew<double>(Eigen::Vector3d(R0.transpose() * a));
    return H;
}
inline Eigen::Matrix<double, 6, 6> stkInvRot(const Eigen::Matrix3d& R0, const Eigen::Vector3d& a,
                                             const Eigen::Vector3d& w) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<3, 3>(0, 0) = se3::skew<double>(a) * se3::skew<double>(Eigen::Vector3d(R0.transpose() * w));
    return H;
}
inline Eigen::Matrix<double, 6, 6> stkInvPoint(const Eigen::Matrix3d& R0, const Eigen::Vector3d& p0,
                                               const Eigen::Vector3d& a) {
    const Eigen::Matrix3d Sa = se3::skew<double>(a);
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<3, 3>(0, 0) = Sa * se3::skew<double>(Eigen::Vector3d(R0.transpose() * p0));
    H.block<3, 3>(0, 3) = Sa;
    return H;
}

Eigen::Matrix<double, 6, 6> sphereHessianFrozen(const Sphere& shape, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector4d& z_soc);

Eigen::Matrix<double, 6, 6> capsuleHessianFrozen(const Capsule& shape, const Eigen::Matrix4d& g0, double t,
                                                  const Eigen::Vector4d& z_soc);

Eigen::Matrix<double, 6, 6> cylinderHessianFrozen(const Cylinder& shape, const Eigen::Matrix4d& g0,
                                                   const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                   const Eigen::Vector4d& z_ort);

Eigen::Matrix<double, 6, 6> coneHessianFrozen(const Cone& shape, const Eigen::Matrix4d& g0, const Eigen::Vector3d& p,
                                               double z_ort, const Eigen::Vector3d& z_soc);

Eigen::Matrix<double, 6, 6> truncatedConeHessianFrozen(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                        const Eigen::Vector3d& p, const Eigen::Vector2d& z_ort,
                                                        const Eigen::Vector3d& z_soc);

Eigen::Matrix<double, 6, 6> ellipsoidHessianFrozen(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                    const Eigen::Vector3d& p, const Eigen::Vector4d& z_soc);

template <int NH>
Eigen::Matrix<double, 6, 6> polytopeHessianFrozen(const Polytope<NH>& shape, const Eigen::Matrix4d& g0,
                                                   const Eigen::Vector3d& p,
                                                   const Eigen::Matrix<double, NH, 1>& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d p0 = g0.block<3, 1>(0, 3);
    // dS = a_poly^T ( d(R^T(R r_offset + p_pose))/dxi - d(R^T p)/dxi ); H = -dS.
    const Eigen::Vector3d a_poly = shape.R_offset * shape.A.transpose() * z_ort;
    return -(stkInvPoint(R0, p0, a_poly) - stkInvRot(R0, a_poly, p));
}

template <int NH>
Eigen::Matrix<double, 6, 6> polygonHessianFrozen(const Polygon<NH>& shape, const Eigen::Matrix4d& g0, double u1,
                                                  double u2, const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    const Eigen::Vector3d Ru = shape.R_offset * Eigen::Vector3d(u1, u2, 0.0);
    // dS = -z_vec^T ( d(R r_offset + p)/dxi + d(R (R_offset u))/dxi ); H = -dS.
    return stkPoint(R0, z_vec, shape.r_offset) + stkRot(R0, z_vec, Ru);
}

// Auto-dispatch, mirroring shapeXiDerivative's overload set below.
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Sphere& shape, const Eigen::Matrix4d& g0,
                                                       const Vec<4>& /*x*/, const Vec<0>& /*z_ort*/,
                                                       const Vec<4>& z_soc) {
    return sphereHessianFrozen(shape, g0, z_soc);
}
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Capsule& shape, const Eigen::Matrix4d& g0, const Vec<5>& x,
                                                       const Vec<2>& /*z_ort*/, const Vec<4>& z_soc) {
    return capsuleHessianFrozen(shape, g0, x(4), z_soc);
}
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Cylinder& shape, const Eigen::Matrix4d& g0,
                                                       const Vec<5>& x, const Vec<4>& z_ort, const Vec<4>& z_soc) {
    return cylinderHessianFrozen(shape, g0, x.head<3>(), x(4), z_soc, z_ort);
}
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Cone& shape, const Eigen::Matrix4d& g0, const Vec<4>& x,
                                                       const Vec<1>& z_ort, const Vec<3>& z_soc) {
    return coneHessianFrozen(shape, g0, x.head<3>(), z_ort(0), z_soc);
}
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                       const Vec<4>& x, const Vec<2>& z_ort, const Vec<3>& z_soc) {
    return truncatedConeHessianFrozen(shape, g0, x.head<3>(), z_ort, z_soc);
}
inline Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                       const Vec<4>& x, const Vec<0>& /*z_ort*/, const Vec<4>& z_soc) {
    return ellipsoidHessianFrozen(shape, g0, x.head<3>(), z_soc);
}
template <int NH>
Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Polytope<NH>& shape, const Eigen::Matrix4d& g0, const Vec<4>& x,
                                                const Vec<NH>& z_ort, const Vec<0>& /*z_soc*/) {
    return polytopeHessianFrozen<NH>(shape, g0, x.head<3>(), z_ort);
}
template <int NH>
Eigen::Matrix<double, 6, 6> shapeHessianFrozen(const Polygon<NH>& shape, const Eigen::Matrix4d& g0, const Vec<6>& x,
                                                const Vec<NH>& /*z_ort*/, const Vec<4>& z_soc) {
    return polygonHessianFrozen<NH>(shape, g0, x(4), x(5), z_soc);
}

// Places shape 2's ShapeXiDerivative into the combined system's dR/dxi
// (dR1_dxi = d(G^T z)/dxi, dR2_dxi = d(cone_product(h-Gx,z))/dxi) and
// q = d(h-Gx)/dxi, for shape 1/2 having any decision-vector widths V1, V2
// (>=4). Shape 1 contributes nothing (fixed pose); only *where* shape 2's
// V2-wide contribution lands in the combined NX-wide vector depends on V1.
// Column layout matches combineProblemMatrices: NX = V1 + (V2-4); shape 2's
// shared [p;alpha] columns stay at 0-3; shape 2's own extras (if V2>4)
// shift to columns [V1, V1+(V2-4)) -- after shape 1's own extras, not
// contiguous with shape 2's first four columns. z is the combined dual
// vector at the converged point.
template <int N_ORT1, int N_SOC1, int N_ORT2, int N_SOC2, int V1, int V2>
struct CombinedXiJacobian {
    static constexpr int nx = V1 + (V2 - 4);
    static constexpr int ns = N_ORT1 + N_ORT2 + N_SOC1 + N_SOC2;
    Eigen::Matrix<double, nx, 6> dR1_dxi = Eigen::Matrix<double, nx, 6>::Zero();
    Eigen::Matrix<double, ns, 6> dR2_dxi = Eigen::Matrix<double, ns, 6>::Zero();
    Eigen::Matrix<double, ns, 6> q = Eigen::Matrix<double, ns, 6>::Zero();
};

template <int N_ORT1, int N_SOC1, int N_ORT2, int N_SOC2, int V1, int V2>
CombinedXiJacobian<N_ORT1, N_SOC1, N_ORT2, N_SOC2, V1, V2> combineXiJacobian(
    const ShapeXiDerivative<N_ORT2, N_SOC2, V2>& shape2_deriv,
    const StackVec<N_ORT1 + N_ORT2, N_SOC1, N_SOC2>& z) {
    CombinedXiJacobian<N_ORT1, N_SOC1, N_ORT2, N_SOC2, V1, V2> out;
    constexpr int ort2_row = N_ORT1;
    constexpr int soc2_row = N_ORT1 + N_ORT2 + N_SOC1;

    // q = d(h - Gx)/dxi: shape 2's ORT/SOC rows only, at its rows in the
    // combined stack.
    if constexpr (N_ORT2 > 0) out.q.template block<N_ORT2, 6>(ort2_row, 0) = shape2_deriv.dHort - shape2_deriv.dGortX;
    if constexpr (N_SOC2 > 0) out.q.template block<N_SOC2, 6>(soc2_row, 0) = shape2_deriv.dHsoc - shape2_deriv.dGsocX;

    // dR1/dxi = d(G^T z)/dxi: shape 2's V2-wide contribution, split and
    // placed per the column layout above.
    const Eigen::Matrix<double, V2, 6> shape2_r1 = shape2_deriv.dGortTz + shape2_deriv.dGsocTz;
    out.dR1_dxi.template topRows<4>() = shape2_r1.template topRows<4>();
    if constexpr (V2 > 4) {
        out.dR1_dxi.template block<V2 - 4, 6>(V1, 0) = shape2_r1.template bottomRows<V2 - 4>();
    }
    // (columns [4, V1) -- shape 1's own extras, if any -- stay zero.)

    // dR2/dxi = d(cone_product(h-Gx,z))/dxi = Arrow(z)*q, z fixed
    // (bilinear); block-diagonal, touches only shape 2's rows.
    if constexpr (N_ORT1 + N_ORT2 > 0) {
        const auto z_ort = z.template head<N_ORT1 + N_ORT2>();
        out.dR2_dxi.template topRows<N_ORT1 + N_ORT2>() =
            z_ort.asDiagonal() * out.q.template topRows<N_ORT1 + N_ORT2>();
    }
    if constexpr (N_SOC1 > 0) {
        const auto z_soc1 = z.template segment<N_SOC1>(N_ORT1 + N_ORT2);
        out.dR2_dxi.template block<N_SOC1, 6>(N_ORT1 + N_ORT2, 0) =
            arrow<N_SOC1>(z_soc1) * out.q.template block<N_SOC1, 6>(N_ORT1 + N_ORT2, 0);
    }
    if constexpr (N_SOC2 > 0) {
        const auto z_soc2 = z.template segment<N_SOC2>(soc2_row);
        out.dR2_dxi.template block<N_SOC2, 6>(soc2_row, 0) =
            arrow<N_SOC2>(z_soc2) * out.q.template block<N_SOC2, 6>(soc2_row, 0);
    }
    return out;
}

// dx/dxi, ds/dxi, dz/dxi at the converged (x,s,z), via the block-eliminated
// IFT solve: A = G'(S\Z)G, dx = A\(r1 - G'\S\r2), dz = (S\Z)G*dx + S\r2,
// ds = q - G*dx. Shape 1 and shape 2 may each carry any decision-vector
// width (>=4). The caller builds shape2_deriv (sphereXiDerivative,
// capsuleXiDerivative, ...); everything else is generic linear algebra.
template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SensitivityResult {
    static constexpr int ns = n_ort + n_soc1 + n_soc2;
    Eigen::Matrix<double, nx, 6> dx;
    Eigen::Matrix<double, ns, 6> ds;
    Eigen::Matrix<double, ns, 6> dz;
};

// The IFT block-elimination proper, given an already-assembled xi_jac
// (combineXiJacobian's output) and the combined G. Split out from
// computeSocpSensitivityWithG so a caller needing both the first-order
// sensitivity and the Hessian (computeContactJacobianBundle) builds
// xi_jac once and runs this solve once.
template <int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
SensitivityResult<n_ort1 + n_ort2, n_soc1, n_soc2, v1 + (v2 - 4)> computeSocpSensitivityFromXiJac(
    const CombinedXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>& xi_jac,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
    const Mat<n_ort1 + n_ort2 + n_soc1 + n_soc2, v1 + (v2 - 4)>& G) {
    constexpr int n_ort = n_ort1 + n_ort2;
    constexpr int ns = n_ort + n_soc1 + n_soc2;
    constexpr int nx = v1 + (v2 - 4);

    const Eigen::Matrix<double, nx, 6> r1 = -xi_jac.dR1_dxi;
    const Eigen::Matrix<double, ns, 6> r2 = -xi_jac.dR2_dxi;

    const PlainScaling<n_ort, n_soc1, n_soc2> Z = plainScalingFromZ<n_ort, n_soc1, n_soc2>(z);
    const NTScaling<n_ort, n_soc1, n_soc2> S = scalingFromS<n_ort, n_soc1, n_soc2>(s);
    const PlainScaling<n_ort, n_soc1, n_soc2> SZ = solve(S, Z); // S \ Z

    const Mat<ns, nx> SZG = SZ.template applyMat<nx>(G);
    const Mat<nx, nx> A = G.transpose() * SZG;
    const Mat<ns, 6> Sr2 = S.template solveMat<6>(r2);
    const Mat<nx, 6> rhs = r1 - G.transpose() * Sr2;

    SensitivityResult<n_ort, n_soc1, n_soc2, nx> out;
    out.dx = A.partialPivLu().solve(rhs);
    out.dz = SZG * out.dx + Sr2;
    out.ds = xi_jac.q - G * out.dx;
    return out;
}

// Same as computeSocpSensitivity below, but takes the combined G
// directly instead of rebuilding it from (shape1,shape2,g0) -- lets a
// caller that already has G (e.g. proximityJacobian, which built it for
// the forward solve) skip reconstructing it for the derivative.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
SensitivityResult<n_ort1 + n_ort2, n_soc1, n_soc2, v1 + (v2 - 4)> computeSocpSensitivityWithG(
    const Shape1& shape1, const Shape2& shape2, const ShapeXiDerivative<n_ort2, n_soc2, v2>& shape2_deriv,
    const Vec<v1 + (v2 - 4)>& x, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z, const Mat<n_ort1 + n_ort2 + n_soc1 + n_soc2, v1 + (v2 - 4)>& G) {
    (void)shape1;
    (void)shape2;
    (void)x;
    const auto xi_jac = combineXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape2_deriv, z);
    return computeSocpSensitivityFromXiJac<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(xi_jac, s, z, G);
}

template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
SensitivityResult<n_ort1 + n_ort2, n_soc1, n_soc2, v1 + (v2 - 4)> computeSocpSensitivity(
    const Shape1& shape1, const Shape2& shape2, const ShapeXiDerivative<n_ort2, n_soc2, v2>& shape2_deriv,
    const Vec<v1 + (v2 - 4)>& x, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z, const Eigen::Matrix4d& g0) {
    constexpr int n_ort = n_ort1 + n_ort2;
    constexpr int ns = n_ort + n_soc1 + n_soc2;
    constexpr int nx = v1 + (v2 - 4);
    const auto P1 = problemMatrices(shape1, Eigen::Matrix4d::Identity());
    const auto P2 = problemMatrices(shape2, g0);
    const auto combined = combineProblemMatrices(P1, P2);
    const Mat<ns, nx>& G = combined.G;
    return computeSocpSensitivityWithG<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, shape2_deriv, x, s, z, G);
}

// Auto-dispatch: one shapeXiDerivative(shape, g0, x, z_ort, z_soc) overload
// per primitive (ordinary overload resolution on the shape's type), so a
// generic caller doesn't need to know which per-shape function applies.
// Unused z_ort/z_soc arguments are simply ignored.

inline ShapeXiDerivative<0, 4, 4> shapeXiDerivative(const Sphere& shape, const Eigen::Matrix4d& g0,
                                                     const Vec<4>& /*x*/, const Vec<0>& /*z_ort*/,
                                                     const Vec<4>& /*z_soc*/) {
    return sphereXiDerivative(shape, g0);
}

inline ShapeXiDerivative<2, 4, 5> shapeXiDerivative(const Capsule& shape, const Eigen::Matrix4d& g0, const Vec<5>& x,
                                                     const Vec<2>& /*z_ort*/, const Vec<4>& z_soc) {
    return capsuleXiDerivative(shape, g0, x(4), z_soc);
}

inline ShapeXiDerivative<4, 4, 5> shapeXiDerivative(const Cylinder& shape, const Eigen::Matrix4d& g0, const Vec<5>& x,
                                                     const Vec<4>& z_ort, const Vec<4>& z_soc) {
    return cylinderXiDerivative(shape, g0, x.head<3>(), x(4), z_soc, z_ort);
}

inline ShapeXiDerivative<2, 3, 4> shapeXiDerivative(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                    const Vec<4>& x, const Vec<2>& z_ort, const Vec<3>& z_soc) {
    return truncatedConeXiDerivative(shape, g0, x.head<3>(), z_ort, z_soc);
}
inline ShapeXiDerivative<1, 3, 4> shapeXiDerivative(const Cone& shape, const Eigen::Matrix4d& g0, const Vec<4>& x,
                                                     const Vec<1>& z_ort, const Vec<3>& z_soc) {
    return coneXiDerivative(shape, g0, x.head<3>(), z_ort(0), z_soc);
}

inline ShapeXiDerivative<0, 4, 4> shapeXiDerivative(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                     const Vec<4>& x, const Vec<0>& /*z_ort*/, const Vec<4>& z_soc) {
    return ellipsoidXiDerivative(shape, g0, x.head<3>(), z_soc);
}

template <int NH>
ShapeXiDerivative<NH, 0, 4> shapeXiDerivative(const Polytope<NH>& shape, const Eigen::Matrix4d& g0, const Vec<4>& x,
                                               const Vec<NH>& z_ort, const Vec<0>& /*z_soc*/) {
    return polytopeXiDerivative<NH>(shape, g0, x.head<3>(), z_ort);
}

template <int NH>
ShapeXiDerivative<NH, 4, 6> shapeXiDerivative(const Polygon<NH>& shape, const Eigen::Matrix4d& g0, const Vec<6>& x,
                                               const Vec<NH>& /*z_ort*/, const Vec<4>& z_soc) {
    return polygonXiDerivative<NH>(shape, g0, x(4), x(5), z_soc);
}

// Shape 2's own x-local (v2-wide) slice, gathered out of the combined
// x = [p;alpha; shape1's extras; shape2's extras]: shared [p;alpha] is
// always x's first 4 entries; shape 2's own extras (if v2>4) sit at
// combined offset v1 -- matching combineXiJacobian's column layout, in
// reverse (gather instead of scatter).
template <int v1, int v2>
Vec<v2> extractShape2LocalX(const Vec<v1 + (v2 - 4)>& x) {
    Vec<v2> out;
    out.template head<4>() = x.template head<4>();
    if constexpr (v2 > 4) {
        out.template tail<v2 - 4>() = x.template segment<v2 - 4>(v1);
    }
    return out;
}

// Same as computeSocpSensitivity, but extracts shape 2's own x_local/
// z_ort/z_soc slices from the combined x, z and dispatches to the right
// shapeXiDerivative overload -- the fully generic entry point.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
SensitivityResult<n_ort1 + n_ort2, n_soc1, n_soc2, v1 + (v2 - 4)> computeSocpSensitivityAuto(
    const Shape1& shape1, const Shape2& shape2, const Vec<v1 + (v2 - 4)>& x,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
    const Eigen::Matrix4d& g0) {
    constexpr int n_ort = n_ort1 + n_ort2;
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    const auto shape2_deriv = shapeXiDerivative(shape2, g0, x2_local, z_ort2, z_soc2);
    return computeSocpSensitivity<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, shape2_deriv, x, s, z, g0);
}

// Auto-dispatch version of computeSocpSensitivityWithG.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
SensitivityResult<n_ort1 + n_ort2, n_soc1, n_soc2, v1 + (v2 - 4)> computeSocpSensitivityAutoWithG(
    const Shape1& shape1, const Shape2& shape2, const Vec<v1 + (v2 - 4)>& x,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
    const Eigen::Matrix4d& g0, const Mat<n_ort1 + n_ort2 + n_soc1 + n_soc2, v1 + (v2 - 4)>& G) {
    constexpr int n_ort = n_ort1 + n_ort2;
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    const auto shape2_deriv = shapeXiDerivative(shape2, g0, x2_local, z_ort2, z_soc2);
    return computeSocpSensitivityWithG<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, shape2_deriv, x, s, z, G);
}

// d(alpha)/dxi (all 6 components), via the envelope-theorem identity
// grad = -q^T z: F_k = z.(G_k x - h_k) = -z.q_k, with q_k := h_k - G_k x
// (the same q combineXiJacobian builds), so stacking k = 0..5 gives
// grad = -q^T z directly.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
Eigen::Matrix<double, 1, 6> computeProximityGradient(const Shape1& shape1, const Shape2& shape2,
                                                       const Vec<v1 + (v2 - 4)>& x,
                                                       const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
                                                       const Eigen::Matrix4d& g0) {
    constexpr int n_ort = n_ort1 + n_ort2;
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    const auto shape2_deriv = shapeXiDerivative(shape2, g0, x2_local, z_ort2, z_soc2);
    const auto xi_jac = combineXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape2_deriv, z);
    return -(xi_jac.q.transpose() * z);
}

// Full 6x6 H_frozen = d(grad)/dxi|_{x,z frozen}: just the shape-2 dispatch,
// which returns the closed-form matrix directly (shape 1 is at the fixed
// reference pose and contributes nothing).
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
Eigen::Matrix<double, 6, 6> hessianFrozenFull(const Shape1& shape1, const Shape2& shape2, const Vec<v1 + (v2 - 4)>& x,
                                               const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
                                               const Eigen::Matrix4d& g0) {
    (void)shape1;
    constexpr int n_ort = n_ort1 + n_ort2;
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    return shapeHessianFrozen(shape2, g0, x2_local, z_ort2, z_soc2);
}

// d^2(alpha)/dxi^2, accounting for how the converged (x*,z*) themselves
// move with xi: grad(xi) = F(xi, x*(xi), z*(xi)), F(xi,x,z) := -q(xi,x)^T z,
// so d(grad)/dxi = dF/dxi|_{x,z fixed} + dF/dx*(dx*/dxi) + dF/dz*(dz*/dxi)
// = H_frozen - r1^T(dx*/dxi) - q^T(dz*/dxi), with dF/dx = -r1^T,
// dF/dz = -q^T (r1 = -dR1_dxi, q = xi_jac.q -- the same objects
// computeSocpSensitivity already builds).
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
Eigen::Matrix<double, 6, 6> computeProximityHessian(const Shape1& shape1, const Shape2& shape2,
                                                      const Vec<v1 + (v2 - 4)>& x,
                                                      const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s,
                                                      const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
                                                      const Eigen::Matrix4d& g0) {
    constexpr int n_ort = n_ort1 + n_ort2;
    constexpr int nx = v1 + (v2 - 4);
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    const auto shape2_deriv = shapeXiDerivative(shape2, g0, x2_local, z_ort2, z_soc2);
    const auto xi_jac = combineXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape2_deriv, z);
    const auto sens = computeSocpSensitivity<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
        shape1, shape2, shape2_deriv, x, s, z, g0);

    const Eigen::Matrix<double, nx, 6> r1 = -xi_jac.dR1_dxi;
    const Eigen::Matrix<double, 6, 6> H_frozen =
        hessianFrozenFull<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape1, shape2, x, z, g0);

    return H_frozen - r1.transpose() * sens.dx - xi_jac.q.transpose() * sens.dz;
}

// d(contact normal)/dxi (proximity_gradient.hpp's contactNormal()).
// n = normalize(u), u = R_g*grad_v (grad_v := grad.tail<3>()) -- product
// rule: R_g depends on xi (dRotatedVectorDXi(g0,grad_v), grad_v frozen)
// *and* grad_v depends on xi via the full re-solved sensitivity (R_g times
// the translational rows of the Hessian above, not H_frozen alone). Then
// the ordinary normalize-derivative chain rule.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
Eigen::Matrix<double, 3, 6> computeContactNormalJacobian(const Shape1& shape1, const Shape2& shape2,
                                                           const Vec<v1 + (v2 - 4)>& x,
                                                           const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s,
                                                           const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
                                                           const Eigen::Matrix4d& g0) {
    const Eigen::Matrix<double, 1, 6> grad =
        computeProximityGradient<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape1, shape2, x, z, g0);
    const Eigen::Vector3d grad_v = grad.template tail<3>().transpose();
    const Eigen::Matrix<double, 6, 6> H =
        computeProximityHessian<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape1, shape2, x, s, z, g0);
    const Eigen::Matrix<double, 3, 6> dgradv_dxi = H.template bottomRows<3>();

    const Eigen::Matrix3d Rg = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d u = Rg * grad_v;
    const double unorm = u.norm();
    const Eigen::Matrix<double, 3, 6> du_dxi = se3::dRotatedVectorDXi(g0, grad_v) + Rg * dgradv_dxi;

    const Eigen::Vector3d n = u / unorm;
    const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - n * n.transpose();
    return (proj / unorm) * du_dxi;
}

// One-shot bundle for proximityContactJacobian's non-degenerate path:
//   jacobian        = d[witness;alpha]/dxi
//   grad            = d(alpha)/dxi
//   normal_jacobian = d(normal)/dxi
// Every shared intermediate is built exactly once -- shape 2's
// xi-derivative, the combined xi_jac (q, dR1/dxi, dR2/dxi), and the
// first-order IFT solve (dx/ds/dz). hessianFrozenFull (one closed-form
// 6x6 H_frozen) is the only work unique to normal_jacobian.
// Bit-identical to computing jacobian/grad/normal_jacobian separately.
template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
struct ContactJacobianBundle {
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
};

template <typename Shape1, typename Shape2, int n_ort1, int n_soc1, int n_ort2, int n_soc2, int v1, int v2>
ContactJacobianBundle<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2> computeContactJacobianBundle(
    const Shape1& shape1, const Shape2& shape2, const Vec<v1 + (v2 - 4)>& x,
    const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& s, const StackVec<n_ort1 + n_ort2, n_soc1, n_soc2>& z,
    const Eigen::Matrix4d& g0, const Mat<n_ort1 + n_ort2 + n_soc1 + n_soc2, v1 + (v2 - 4)>& G) {
    constexpr int n_ort = n_ort1 + n_ort2;
    constexpr int nx = v1 + (v2 - 4);

    // --- shared intermediates, each built once ---
    const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(x);
    const Vec<n_ort2> z_ort2 = z.template segment<n_ort2>(n_ort1);
    const Vec<n_soc2> z_soc2 = z.template segment<n_soc2>(n_ort + n_soc1);
    const auto shape2_deriv = shapeXiDerivative(shape2, g0, x2_local, z_ort2, z_soc2);
    const auto xi_jac = combineXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape2_deriv, z);
    const auto sens = computeSocpSensitivityFromXiJac<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(xi_jac, s, z, G);

    ContactJacobianBundle<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2> out;
    out.jacobian = sens.dx.template topRows<4>();
    out.grad = -(xi_jac.q.transpose() * z); // envelope theorem: d(alpha)/dxi = -q^T z

    // --- d^2(alpha)/dxi^2: hessianFrozenFull is the only new work; the two
    //     IFT correction terms reuse sens + xi_jac (cf. computeProximityHessian) ---
    const Eigen::Matrix<double, nx, 6> r1 = -xi_jac.dR1_dxi;
    const Eigen::Matrix<double, 6, 6> H_frozen =
        hessianFrozenFull<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape1, shape2, x, z, g0);
    const Eigen::Matrix<double, 6, 6> H = H_frozen - r1.transpose() * sens.dx - xi_jac.q.transpose() * sens.dz;

    // --- d(normal)/dxi: normalize-chain on grad + H's translational rows,
    //     identical to computeContactNormalJacobian ---
    const Eigen::Vector3d grad_v = out.grad.template tail<3>().transpose();
    const Eigen::Matrix<double, 3, 6> dgradv_dxi = H.template bottomRows<3>();
    const Eigen::Matrix3d Rg = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d u = Rg * grad_v;
    const double unorm = u.norm();
    const Eigen::Matrix<double, 3, 6> du_dxi = se3::dRotatedVectorDXi(g0, grad_v) + Rg * dgradv_dxi;
    const Eigen::Vector3d nrm = u / unorm;
    const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - nrm * nrm.transpose();
    out.normal_jacobian = (proj / unorm) * du_dxi;

    return out;
}

} // namespace dcolpp::socp
