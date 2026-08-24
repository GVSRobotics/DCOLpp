// Definitions for dcolpp/se3.hpp's plain-double functions.
#include "dcolpp/se3.hpp"

#include <cmath>

namespace dcolpp::se3 {

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

Matrix6d adjoint_se3(const Vector6d& xi) {
    Matrix6d ad = Matrix6d::Zero();
    const Matrix3d W = skew<double>(xi.head<3>());
    ad.block<3, 3>(0, 0) = W;
    ad.block<3, 3>(3, 3) = W;
    ad.block<3, 3>(3, 0) = skew<double>(xi.tail<3>());
    return ad;
}

Matrix6d tangent_se3(const Vector6d& xi) {
    const double theta2 = xi.head<3>().squaredNorm();
    const Matrix6d ad = adjoint_se3(xi);
    const Matrix6d ad2 = ad * ad;
    const Matrix6d ad3 = ad2 * ad;
    const Matrix6d ad4 = ad3 * ad;

    if (theta2 <= kThetaThresholdSq) {
        return Matrix6d::Identity() + 0.5 * ad + (1.0 / 6.0) * ad2 + (1.0 / 24.0) * ad3 + (1.0 / 120.0) * ad4;
    }
    const double theta = std::sqrt(theta2);
    const double c = std::cos(theta), s = std::sin(theta);
    const double t3 = theta2 * theta, t4 = t3 * theta, t5 = t4 * theta;
    const double c1 = (4.0 - 4.0 * c - theta * s) / (2.0 * theta2);
    const double c2 = (4.0 * theta - 5.0 * s + theta * c) / (2.0 * t3);
    const double c3 = (2.0 - 2.0 * c - theta * s) / (2.0 * t4);
    const double c4 = (2.0 * theta - 3.0 * s + theta * c) / (2.0 * t5);
    return Matrix6d::Identity() + c1 * ad + c2 * ad2 + c3 * ad3 + c4 * ad4;
}

Matrix6d tangentDot_se3(const Vector6d& xi, const Vector6d& xi_dot) {
    const double theta2 = xi.head<3>().squaredNorm();
    const Matrix6d ad = adjoint_se3(xi);
    const Matrix6d ad_dot = adjoint_se3(xi_dot);
    const Matrix6d ad2 = ad * ad;
    const Matrix6d ad3 = ad2 * ad;
    const Matrix6d ad4 = ad3 * ad;
    const Matrix6d ad_dot2 = ad_dot * ad + ad * ad_dot;
    const Matrix6d ad_dot3 = ad_dot2 * ad + ad2 * ad_dot;
    const Matrix6d ad_dot4 = ad_dot3 * ad + ad3 * ad_dot;

    if (theta2 <= kThetaThresholdSq) {
        return 0.5 * ad_dot + (1.0 / 6.0) * ad_dot2 + (1.0 / 24.0) * ad_dot3 + (1.0 / 120.0) * ad_dot4;
    }
    const double theta = std::sqrt(theta2);
    const double theta_dot = xi_dot.head<3>().dot(xi.head<3>()) / theta;
    const double c = std::cos(theta), s = std::sin(theta);
    const double t2 = theta2, t3 = t2 * theta, t4 = t3 * theta, t5 = t4 * theta, t6 = t5 * theta;

    const double term1 = -8.0 + (8.0 - t2) * c + 5.0 * theta * s;
    const double term2 = 4.0 - 4.0 * c - theta * s;
    const double term3 = -8.0 * theta + (15.0 - t2) * s - 7.0 * theta * c;
    const double term4 = 4.0 * theta - 5.0 * s + theta * c;
    const double term5 = 2.0 - 2.0 * c - theta * s;
    const double term6 = 2.0 * theta - 3.0 * s + theta * c;

    return theta_dot * (0.5 / t3) * term1 * ad + (0.5 / t2) * term2 * ad_dot +
           theta_dot * (0.5 / t4) * term3 * ad2 + (0.5 / t3) * term4 * ad_dot2 +
           theta_dot * (0.5 / t5) * term1 * ad3 + (0.5 / t4) * term5 * ad_dot3 +
           theta_dot * (0.5 / t6) * term3 * ad4 + (0.5 / t5) * term6 * ad_dot4;
}

Matrix6d tangentRight(const Vector6d& xi) { return tangent_se3(-xi); }

Matrix6d tangentDotRight(const Vector6d& xi, const Vector6d& xi_dot) { return tangentDot_se3(-xi, -xi_dot); }

Eigen::Matrix<double, 3, 6> dRotatedVectorDXi(const Matrix4d& g0, const Vector3d& v_local) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(v_local);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

Eigen::Matrix<double, 3, 6> dPointDXi(const Matrix4d& g0, const Vector3d& r_local) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(r_local);
    out.block<3, 3>(0, 3) = R0;
    return out;
}

Eigen::Matrix<double, 3, 6> dInverseRotatedVectorDXi(const Matrix4d& g0, const Vector3d& w) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = skew<double>(R0.transpose() * w);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

Eigen::Matrix<double, 3, 6> d2PointDXi(const Matrix4d& g0, const Vector3d& r_local, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    const Matrix3d Sdw = skew<double>(dw);
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * Sdw * skew<double>(r_local);
    out.block<3, 3>(0, 3) = R0 * Sdw;
    return out;
}

Eigen::Matrix<double, 3, 6> d2RotatedVectorDXi(const Matrix4d& g0, const Vector3d& v_local, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -R0 * skew<double>(dw) * skew<double>(v_local);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

Eigen::Matrix<double, 3, 6> d2InverseRotatedVectorDXi(const Matrix4d& g0, const Vector3d& w, const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    Eigen::Matrix<double, 3, 6> out;
    out.block<3, 3>(0, 0) = -skew<double>(skew<double>(dw) * R0.transpose() * w);
    out.block<3, 3>(0, 3).setZero();
    return out;
}

Eigen::Matrix<double, 3, 6> d2InverseRotatedPointDXi(const Matrix4d& g0, const Vector3d& r_local,
                                                      const Vector6d& d) {
    const Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Vector3d dw = d.head<3>();
    const Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * r_local;
    const Eigen::Matrix<double, 3, 6> dP_g0 = dPointDXi(g0, r_local);
    const Vector3d dr_dt = dP_g0 * d;

    Eigen::Matrix<double, 3, 6> term2 = Eigen::Matrix<double, 3, 6>::Zero();
    term2.block<3, 3>(0, 0) = skew<double>(R0.transpose() * dr_dt);

    return d2InverseRotatedVectorDXi(g0, r0, d) + term2 - skew<double>(dw) * R0.transpose() * dP_g0 +
           R0.transpose() * d2PointDXi(g0, r_local, d);
}

} // namespace dcolpp::se3
