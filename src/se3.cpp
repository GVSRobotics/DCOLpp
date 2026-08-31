// Definitions for dcolpp/se3.hpp's plain-double functions.
#include "dcolpp/se3.hpp"

#include <cmath>

namespace dcolpp::se3 {

// Conventions for the derivative functions.
//   SE(3) pose: g0 = (R0, p0); twist xi = [w; v]; g(xi) = g0 * Exp(xi).
//   Exp(xi) = [ ExpSO3(w)  V(w)*v ;  0  1 ], V = left SO(3) Jacobian, V(0)=I.
//   Rotations: R(xi) = R0 * ExpSO3(w) exactly.
//   Positions: p(xi) = p0 + R0*V(w)*v,  with  dp_dw|_0 = 0,  dp_dv|_0 = R0.
//   All derivatives are at xi = 0.  a_tilde = skew(a) (the so(3) hat);
//   d/dw_k ExpSO3(w)|_0 = e_k_tilde.
//
// Two derivative kinds for a vector quantity f(g).  d is a CONSTANT
// direction 6-vector; dw := d.head<3>().
//
//   df_dxi = [ df_dw | df_dv ]   the 3x6 Jacobian of f w.r.t. the pose
//     twist, at xi = 0  (dRotatedVectorDXi / dPointDXi /
//     dInverseRotatedVectorDXi).
//
//   d2f(g0, ., d)   the derivative of the pose-field  g |-> df_dxi(g)  in
//     the pose-twist direction d -- a 3x6 matrix.  d says which direction
//     of xi to differentiate along; it is NOT contracted into df_dxi.
//     Downstream (shapeHessianFrozen) forms  z_vec^T * d2f, i.e.
//     d_dxi[ z_vec^T * df_dxi ] -- the frozen-dual contraction of the
//     Jacobian, differentiated w.r.t. the pose; calling with d = e_0..e_5
//     builds the 6 columns of the 6x6 H_frozen.  Because df_dxi is R (or
//     R^T) times constants, d2f just replaces R by its derivative
//     d_dxi(R) * d = R0 * dw_tilde -- so for d2PointDXi / d2RotatedVectorDXi
//     / d2InverseRotatedVectorDXi only dw enters; d2InverseRotatedPointDXi's
//     f carries a placed point r(xi), so all of d enters.
//
// Body-twist principle: d_dxi[ g0*Exp(xi) ]|_0 has columns g0 * e_k_hat
// (e_k_hat the 4x4 se(3) hat), i.e. dg = g0 * [ dw_tilde  dv ; 0  0 ] -- this
// IS the body-twist variation of g, not an approximation, and holds for any
// retraction, not just Exp. Hence for a contact quantity h(g),
//   dh_dq = (dh_dxi at xi = 0) * J_rel(q)      (no Ad_g between the factors)
// since the robot body Jacobian is defined the same way,
// (g^{-1} g_dot)^v = J q_dot, i.e. dg = g * [ (J dq)_w tilde  (J dq)_v ; 0 0 ].
// J_rel = J2_body if body 1 is world-fixed, else J2_body - Ad_{g^{-1}} J1_body.

// g^{-1} = [ R^T | -R^T p ].
Matrix4d SE3Inverse(const Matrix4d& g) {
    Matrix4d out = Matrix4d::Identity();
    const Matrix3d Rt = g.block<3, 3>(0, 0).transpose();
    out.block<3, 3>(0, 0) = Rt;
    out.block<3, 1>(0, 3) = -Rt * g.block<3, 1>(0, 3);
    return out;
}

Matrix4d relative(const Matrix4d& g1, const Matrix4d& g2) {
    return SE3Inverse(g1) * g2;
}

// f = R * v_local  (a direction fixed in g0's frame).
//   df_dw = d_dw [R0*ExpSO3(w)*v_local] = -R0 * v_local_tilde
//   df_dv = 0   (a direction does not translate)
Eigen::Matrix<double, 3, 6> dRotatedVectorDXi(const Matrix4d& g0, const Vector3d& v_local) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(v_local);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

// f = R*r_local + p  (a point fixed in g0's frame).
//   df_dw = -R0 * r_local_tilde
//   df_dv =  R0     (= dp_dv|_0, see preamble)
Eigen::Matrix<double, 3, 6> dPointDXi(const Matrix4d& g0, const Vector3d& r_local) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(r_local);
    out.block<3, 3>(0, 3) = R0;
    return out;
}

// f = R^T * a  for a FIXED reference-frame vector a (the parameter `w`, not
// the twist's w). R(xi)^T = ExpSO3(-w_twist), so
//   df_dw_twist = d_dw_twist [ExpSO3(-w_twist) * R0^T * a] = (R0^T * a)_tilde
//   df_dv = 0
// Opposite sign to dPointDXi -- a sits on the un-rotated side of R.
Eigen::Matrix<double, 3, 6> dInverseRotatedVectorDXi(const Matrix4d& g0, const Vector3d& w) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = skew<double>(R0.transpose() * w);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

// d2f = derivative of the field  g |-> dPointDXi(g)  in pose direction d.
// dPointDXi(g) = [ -R*r_local_tilde | R ] is R times constants, so replace
// R by d_dxi(R) * d = R0 * dw_tilde (dw := d.head<3>()):
//   d2f_dw = -R0 * dw_tilde * r_local_tilde
//   d2f_dv =  R0 * dw_tilde
Eigen::Matrix<double, 3, 6> d2PointDXi(const Matrix4d& g0, const Vector3d& r_local, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    const Matrix3d Sdw = skew<double>(dw);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * Sdw * skew<double>(r_local);
    out.block<3, 3>(0, 3) = R0 * Sdw;
    return out;
}

// d2f = derivative of  g |-> dRotatedVectorDXi(g) = [ -R*v_local_tilde | 0 ]
// in pose direction d; replace R by R0 * dw_tilde:
//   d2f_dw = -R0 * dw_tilde * v_local_tilde
//   d2f_dv = 0
Eigen::Matrix<double, 3, 6> d2RotatedVectorDXi(const Matrix4d& g0, const Vector3d& v_local, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(dw) * skew<double>(v_local);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

// d2f = derivative of  g |-> dInverseRotatedVectorDXi(g) = [ (R^T*a)_tilde | 0 ]
// in pose direction d, a a FIXED reference vector; replace R^T by its
// derivative d_dxi(R^T) * d = -dw_tilde * R0^T:
//   d2f_dw = -( dw_tilde * R0^T * a )_tilde
//   d2f_dv = 0
Eigen::Matrix<double, 3, 6> d2InverseRotatedVectorDXi(const Matrix4d& g0, const Vector3d& w, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -skew<double>(skew<double>(dw) * R0.transpose() * w);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

// f = R^T * r(xi) with r(xi) = R*r_local + p itself a PLACED point (the
// R^T*r pattern in Cone/Ellipsoid/Polytope h_soc/h_ort). Product rule:
//   df_dxi = dInverseRotatedVectorDXi(g, r) + R^T * dPointDXi(g, r_local),
// so d2f = derivative of that field in pose direction d.  The inner r
// co-moves as the placed point (its own velocity dr = dPointDXi(g0,r_local)
// * d), so here all of d enters. With r0 = p0 + R0*r_local it is
//   d2InverseRotatedVectorDXi(g0, r0, d)         (g varies, r frozen at r0)
//   + ( R0^T * dr )_tilde                        (r varies, term 1)
//   - dw_tilde * R0^T * dPointDXi(g0, r_local)   (R^T varies, term 2)
//   + R0^T * d2PointDXi(g0, r_local, d)          (dPointDXi varies, term 2)
Eigen::Matrix<double, 3, 6> d2InverseRotatedPointDXi(const Matrix4d& g0, const Vector3d& r_local,
                                                      const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    const Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * r_local;
    const Eigen::Matrix<double, 3, 6> dP_g0 = dPointDXi(g0, r_local);
    const Vector3d dr = dP_g0 * d;

    Eigen::Matrix<double, 3, 6> term2 = Eigen::Matrix<double, 3, 6>::Zero();
    term2.block<3, 3>(0, 0) = skew<double>(R0.transpose() * dr);

    return d2InverseRotatedVectorDXi(g0, r0, d) + term2 - skew<double>(dw) * R0.transpose() * dP_g0 +
           R0.transpose() * d2PointDXi(g0, r_local, d);
}

} // namespace dcolpp::se3
