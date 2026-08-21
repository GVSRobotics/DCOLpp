#pragma once
// dcolpp::se3 — SE(3) helpers shared by both engines.
//
// Pose convention: a relative pose is a plain `Eigen::Matrix4d g` such that
// a point expressed in body 2's local
// frame maps into body 1's frame (the reference frame for a proximity pair)
// via `x1 = g.linear()*y2 + g.translation()`. `g = g1^{-1} g2` if the two
// bodies additionally have their own absolute world poses g1, g2.
//
// Twist convention: xi = [w;v] in R^6 -- rotation (w, indices 0-2) first,
// translation (v, indices 3-5) second. DCOL++ always composes a twist by
// RIGHT multiplication onto a LOCAL frame: g(xi) = g0 * Exp(xi).
//
// Plain `double` functions (SE3Inverse, relative, adjoint_se3, tangent_se3,
// tangentDot_se3, tangentRight, tangentDotRight) are declared here and
// defined in src/se3.cpp. The templated ones (skew, hat, Exp, retract) stay
// header-only: template code can't be split into a .cpp without explicit
// instantiation, and here T ranges over more than a fixed pair of types.

#include <Eigen/Dense>

#include "dcolpp/dual6.hpp"

namespace dcolpp::se3 {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

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

// SE(3) exponential map, as a single hat-matrix power series
// g = I + H + c2*H^2 + c3*H^3. Generic over scalar type T, deduced from any
// Eigen expression (e.g. `-e`)
// via MatrixBase<Derived> rather than a concrete Matrix<T,6,1> parameter,
// since template deduction doesn't apply implicit conversions.
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Exp(const Eigen::MatrixBase<Derived>& xi_in) {
    using T = typename Derived::Scalar;
    const Eigen::Matrix<T, 6, 1> xi = xi_in;
    // theta^2, not theta itself: theta = sqrt(theta2) has an undefined
    // derivative at theta = 0, exactly where autodiff evaluates this (xi
    // seeded at 0), so the small-angle branch is written entirely in
    // theta2 -- smooth everywhere -- and never calls sqrt.
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

// g(xi) = g0 * Exp(xi), differentiated at xi = 0. `g0` is the current
// (double) evaluation point; `xi` carries whatever scalar type (double,
// Dual6) the caller wants derivatives in.
template <typename T>
Eigen::Matrix<T, 4, 4> retract(const Matrix4d& g0, const Eigen::Matrix<T, 6, 1>& xi) {
    return g0.template cast<T>() * Exp(xi);
}

// The six unit-seeded Dual6 directions for xi at value 0. Feed into
// `retract<Dual6>(g0, seedTwist())` to get a Dual6-typed g whose `.grad()`
// on any downstream scalar is the Jacobian w.r.t. that twist.
inline Eigen::Matrix<Dual6, 6, 1> seedTwist() {
    Eigen::Matrix<Dual6, 6, 1> xi;
    for (int i = 0; i < 6; ++i) xi(i) = Dual6::seed(0.0, i);
    return xi;
}

// se(3) adjoint operator ad_v = [skew(w), 0; skew(v), skew(w)], xi=[w;v]
// (the generator of the Lie bracket [xi, .]).
Matrix6d adjoint_se3(const Vector6d& xi);

// The LEFT-trivialized Jacobian of Exp at xi (not just at xi=0):
// Exp(xi+dxi) ~= Exp(T(xi)*dxi) * Exp(xi), via the closed-form series
// T(xi) = I + c1*ad_xi + c2*ad_xi^2 + c3*ad_xi^3 + c4*ad_xi^4 (a standard
// Lie-group result, not derived by differentiating Exp's own formula).
// DCOL++ itself always uses RIGHT multiplication with a LOCAL twist
// (g0*Exp(xi)) -- use `tangentRight`, not this, anywhere in this codebase.
Matrix6d tangent_se3(const Vector6d& xi);

// d/dt[tangent_se3(xi(t))] for xi moving along xi_dot -- a directional
// (Jacobian-vector-product) second derivative, not a full Hessian tensor.
Matrix6d tangentDot_se3(const Vector6d& xi, const Vector6d& xi_dot);

// The RIGHT-trivialized (local-frame) Jacobian of Exp -- the convention
// DCOL++ uses throughout: Exp(xi+dxi) ~= Exp(xi) * Exp(tangentRight(xi)*dxi).
// J_right(xi) = J_left(-xi) (standard identity).
Matrix6d tangentRight(const Vector6d& xi);
Matrix6d tangentDotRight(const Vector6d& xi, const Vector6d& xi_dot);

} // namespace dcolpp::se3
