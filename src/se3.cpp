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
// For a vector quantity f(g), df_dxi = [ df_dw | df_dv ] is its 3x6 Jacobian
// w.r.t. the pose twist, at xi = 0 (dRotatedVectorDXi / dPointDXi /
// dInverseRotatedVectorDXi).
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

} // namespace dcolpp::se3
