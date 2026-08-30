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
