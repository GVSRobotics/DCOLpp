#pragma once
// dcolpp::se3 -- SE(3) exponential map and derivatives of points/vectors
// placed by a pose, w.r.t. a 6-dof local twist.
//
// Pose: `Eigen::Matrix4d g` maps a point in body 2's local frame into body
// 1's frame, `x1 = g.linear()*y2 + g.translation()` (g = g1^{-1} g2 for two
// bodies with their own world poses).
//
// Twist: xi = [w;v] in R^6 (rotation first, translation second), composed
// by RIGHT multiplication onto a LOCAL frame: g(xi) = g0 * Exp(xi).
//
// Plain `double` functions are declared here, defined in src/se3.cpp.
// `skew`, `hat`, `Exp`, `retract` stay header-only (scalar-type templates).

#include <Eigen/Dense>

namespace dcolpp::se3 {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

constexpr double kThetaThreshold = 1.0e-2;
constexpr double kThetaThresholdSq = kThetaThreshold * kThetaThreshold;

// so(3) hat operator.
template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& w) {
    Eigen::Matrix<T, 3, 3> S;
    S <<    T(0), -w(2),  w(1),
            w(2),  T(0), -w(0),
           -w(1),  w(0),  T(0);
    return S;
}

// se(3) hat operator: [skew(w), v; 0, 0], xi = [w;v].
template <typename T>
Eigen::Matrix<T, 4, 4> hat(const Eigen::Matrix<T, 6, 1>& xi) {
    Eigen::Matrix<T, 4, 4> H = Eigen::Matrix<T, 4, 4>::Zero();
    H.template block<3, 3>(0, 0) = skew<T>(xi.template head<3>());
    H.template block<3, 1>(0, 3) = xi.template tail<3>();
    return H;
}

// SE(3) exponential map: g = I + H + c2*H^2 + c3*H^3. Scalar type T deduced
// via MatrixBase<Derived> so expressions like `Exp(-e)` work directly.
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Exp(const Eigen::MatrixBase<Derived>& xi_in) {
    using T = typename Derived::Scalar;
    const Eigen::Matrix<T, 6, 1> xi = xi_in;
    // theta^2, not theta: keeps the small-angle branch free of sqrt (whose
    // derivative is undefined at 0).
    const T theta2 = xi.template head<3>().squaredNorm();
    const Eigen::Matrix<T, 4, 4> H = hat<T>(xi);
    const Eigen::Matrix<T, 4, 4> H2 = H * H;
    const Eigen::Matrix<T, 4, 4> H3 = H2 * H;

    T c2, c3;
    if (theta2 <= T(kThetaThresholdSq)) {
        c2 = T(0.5) - theta2 / T(24.0);
        c3 = T(1.0 / 6.0) - theta2 / T(120.0);
    } else {
        using std::sqrt;
        const T theta = sqrt(theta2);
        c2 = (T(1.0) - cos(theta)) / theta2;
        c3 = (theta - sin(theta)) / (theta2 * theta);
    }
    return Eigen::Matrix<T, 4, 4>::Identity() + H + c2 * H2 + c3 * H3;
}

Matrix4d SE3Inverse(const Matrix4d& g);
Matrix4d relative(const Matrix4d& g1, const Matrix4d& g2); // g1^{-1} g2

// g(xi) = g0 * Exp(xi). `g0` is the (double) evaluation point; `xi` carries
// whatever scalar type the caller wants (e.g. an autodiff type).
template <typename T>
Eigen::Matrix<T, 4, 4> retract(const Matrix4d& g0, const Eigen::Matrix<T, 6, 1>& xi) {
    return g0.template cast<T>() * Exp(xi);
}

// d(R0*v_local)/dxi at xi=0, g(xi)=g0*Exp(xi), g0=(R0,p0). v_local is a
// direction fixed in g0's frame (e.g. a shape's local axis): rotation
// columns are -R0*skew(v_local), translation columns are zero.
Eigen::Matrix<double, 3, 6> dRotatedVectorDXi(const Matrix4d& g0, const Vector3d& v_local);

// d(R0*r_local + p0)/dxi at xi=0, for a point fixed in g0's frame: rotation
// columns are -R0*skew(r_local), translation columns are R0.
Eigen::Matrix<double, 3, 6> dPointDXi(const Matrix4d& g0, const Vector3d& r_local);

// d(R0^T * w)/dxi at xi=0, for a fixed world-frame vector w. Rotation
// columns are skew(R0^T*w) (sign opposite dPointDXi/dRotatedVectorDXi:
// w sits on the un-rotated side); translation columns are zero. If w itself
// varies with xi, combine with dPointDXi via the product rule (see
// coneXiDerivative's h_soc).
Eigen::Matrix<double, 3, 6> dInverseRotatedVectorDXi(const Matrix4d& g0, const Vector3d& w);

} // namespace dcolpp::se3
