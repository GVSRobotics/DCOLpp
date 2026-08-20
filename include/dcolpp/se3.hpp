#pragma once
// dcolpp::se3 — SE(3) helpers shared by both engines.
//
// Pose convention (matches iDCOL's `ProblemData::g`): a relative pose is a
// plain `Eigen::Matrix4d g` such that a point expressed in body 2's local
// frame maps into body 1's frame (the reference frame for a proximity pair)
// via `x1 = g.linear()*y2 + g.translation()`. `g = g1^{-1} g2` if the two
// bodies additionally have their own absolute world poses g1, g2.
//
// Two ways to move g are provided, deliberately different:
//
//  - `Exp` / `SE3Compose` / `SE3Inverse` — exact SE(3) composition, for
//    actually moving a body (examples, integrators, building a relative
//    pose out of two absolute ones). Always operates on `double`.
//
//  - `retract<T>` — the perturbation used internally by both engines'
//    differentiation: a FIRST-ORDER local retraction of g by a right twist
//    xi = [v;w] (i.e. g(xi) ~= g0 * Exp(xi), to O(|xi|^2)). It is templated
//    on scalar type so it can be seeded with `Dual6` and differentiated at
//    xi = 0. Any valid retraction has the same derivative as the true
//    exponential map at the base point, so using the cheap linear form here
//    costs nothing in the resulting Jacobian/gradient while sidestepping
//    the removable sin(theta)/theta singularity an exact-exp implementation
//    would need to special-case at theta = 0 (exactly where xi is always
//    evaluated, since we differentiate the family g0 * Exp(xi) at xi = 0).

#include <Eigen/Dense>
#include <cmath>

#include "dcolpp/dual6.hpp"

namespace dcolpp::se3 {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// so(3) hat operator, templated so it can be used with Dual6 too.
template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& w) {
    Eigen::Matrix<T, 3, 3> S;
    S <<    T(0), -w(2),  w(1),
            w(2),  T(0), -w(0),
           -w(1),  w(0),  T(0);
    return S;
}

// Exact SO(3)/SE(3) exponential map (Rodrigues / the standard SE(3) closed
// form), double precision only. Used to move bodies by a finite twist and
// as independent ground truth for testing `retract`'s derivative.
inline Matrix4d Exp(const Vector6d& xi) {
    const Vector3d v = xi.head<3>();
    const Vector3d w = xi.tail<3>();
    const double theta = w.norm();

    const Matrix3d W = skew<double>(w);
    double a, b; // R = I + a*W + b*W^2 ; V = I + b*W + c*W^2 (c below)
    double c;
    if (theta < 1e-8) {
        // Taylor series around theta = 0 (removable sin/cos singularities).
        a = 1.0 - theta * theta / 6.0;
        b = 0.5 - theta * theta / 24.0;
        c = 1.0 / 6.0 - theta * theta / 120.0;
    } else {
        const double t2 = theta * theta;
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / t2;
        c = (theta - std::sin(theta)) / (theta * t2);
    }

    const Matrix3d W2 = W * W;
    Matrix3d R = Matrix3d::Identity() + a * W + b * W2;
    Matrix3d V = Matrix3d::Identity() + b * W + c * W2;

    Matrix4d g = Matrix4d::Identity();
    g.block<3, 3>(0, 0) = R;
    g.block<3, 1>(0, 3) = V * v;
    return g;
}

inline Matrix4d SE3Inverse(const Matrix4d& g) {
    Matrix4d out = Matrix4d::Identity();
    const Matrix3d Rt = g.block<3, 3>(0, 0).transpose();
    out.block<3, 3>(0, 0) = Rt;
    out.block<3, 1>(0, 3) = -Rt * g.block<3, 1>(0, 3);
    return out;
}

inline Matrix4d SE3Compose(const Matrix4d& g1, const Matrix4d& g2) {
    return g1 * g2;
}

// Relative transform from two absolute poses: g = g1^{-1} g2.
inline Matrix4d relative(const Matrix4d& g1, const Matrix4d& g2) {
    return SE3Compose(SE3Inverse(g1), g2);
}

// First-order local retraction: g(xi) ~= g0 * Exp(xi), differentiated at
// xi = 0. `g0` is always the current (double) evaluation point; `xi` carries
// whatever scalar type (double or Dual6) the caller wants derivatives in.
template <typename T>
Eigen::Matrix<T, 4, 4> retract(const Matrix4d& g0, const Eigen::Matrix<T, 6, 1>& xi) {
    const Eigen::Matrix<T, 3, 1> v = xi.template head<3>();
    const Eigen::Matrix<T, 3, 1> w = xi.template tail<3>();

    const Eigen::Matrix<T, 3, 3> R0 = g0.block<3, 3>(0, 0).template cast<T>();
    const Eigen::Matrix<T, 3, 1> p0 = g0.block<3, 1>(0, 3).template cast<T>();

    const Eigen::Matrix<T, 3, 3> Omega = skew<T>(w);
    const Eigen::Matrix<T, 3, 3> R = R0 * (Eigen::Matrix<T, 3, 3>::Identity() + Omega);
    const Eigen::Matrix<T, 3, 1> p = p0 + R0 * v;

    Eigen::Matrix<T, 4, 4> g = Eigen::Matrix<T, 4, 4>::Identity();
    g.template block<3, 3>(0, 0) = R;
    g.template block<3, 1>(0, 3) = p;
    return g;
}

// The six unit-seeded Dual6 directions for xi = [v;w] at value 0 -- i.e.
// "differentiate w.r.t. the local twist of g at the current pose". Feed the
// result into `retract<Dual6>(g0, seedTwist())` to get a Dual6-typed g whose
// `.grad()` on any downstream scalar is the Jacobian w.r.t. that twist.
inline Eigen::Matrix<Dual6, 6, 1> seedTwist() {
    Eigen::Matrix<Dual6, 6, 1> xi;
    for (int i = 0; i < 6; ++i) xi(i) = Dual6::seed(0.0, i);
    return xi;
}

} // namespace dcolpp::se3
