#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/primitives.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Geometric primitives. Unlike the Julia source (which gives every shape a
// world-frame position/quaternion pair, duplicated again for an MRP
// variant), DCOL++ primitives carry only local geometry: placement is
// entirely handled by the relative SE(3) pose `g` passed into
// `problemMatrices` (see problem_matrices.hpp), so there is exactly one
// struct per shape family instead of a quaternion/MRP pair.
//
// `r_offset`/`Q_offset` (a local mounting translation/rotation) are kept,
// matching the original -- they let a primitive's collision geometry be
// offset from the frame origin it's attached to.

#include <Eigen/Dense>

namespace dcolpp::socp {

struct Capsule {
    double R, L;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    Capsule(double R_, double L_) : R(R_), L(L_) {}
};

struct Cylinder {
    double R, L;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    Cylinder(double R_, double L_) : R(R_), L(L_) {}
};

struct Cone {
    double H, beta;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    Cone(double H_, double beta_) : H(H_), beta(beta_) {}
};

struct Sphere {
    double R;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    explicit Sphere(double R_) : R(R_) {}
};

struct Ellipsoid {
    Eigen::Matrix3d P;  // x'Px <= 1
    Eigen::Matrix3d U;  // upper Cholesky factor of P
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    explicit Ellipsoid(const Eigen::Matrix3d& P_) : P(P_) {
        U = Eigen::LLT<Eigen::Matrix3d>(P_).matrixU();
    }
};

template <int NH>
struct Polytope {
    Eigen::Matrix<double, NH, 3> A;
    Eigen::Matrix<double, NH, 1> b;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    Polytope(const Eigen::Matrix<double, NH, 3>& A_, const Eigen::Matrix<double, NH, 1>& b_) : A(A_), b(b_) {}
};

template <int NH>
struct Polygon {
    Eigen::Matrix<double, NH, 2> A;
    Eigen::Matrix<double, NH, 1> b;
    double R; // "cushion" radius
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    Polygon(const Eigen::Matrix<double, NH, 2>& A_, const Eigen::Matrix<double, NH, 1>& b_, double R_)
        : A(A_), b(b_), R(R_) {}
};

} // namespace dcolpp::socp
