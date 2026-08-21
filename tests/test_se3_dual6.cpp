#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <vector>

#include "dcolpp/dual6.hpp"
#include "dcolpp/se3.hpp"

using namespace dcolpp;
using Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomPose(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    // start from a random-ish base pose reached by a finite (non-small) twist
    xi.head<3>() *= 0.8; // keep rotation part well away from wrap-around edge cases
    Matrix4d g = se3::Exp(xi);
    g.block<3, 1>(0, 3) += Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

bool isOrthonormal(const Eigen::Matrix3d& R, double tol = 1e-9) {
    return (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < tol;
}

} // namespace

TEST_CASE("Dual6: basic arithmetic matches hand derivatives", "[dual6]") {
    Dual6 x = Dual6::seed(2.0, 0);
    Dual6 y = Dual6::seed(3.0, 1);

    Dual6 s = x + y;
    REQUIRE(s.value() == 5.0);
    REQUIRE(s.grad()(0) == 1.0);
    REQUIRE(s.grad()(1) == 1.0);

    Dual6 p = x * y; // d(xy)/dx = y, d(xy)/dy = x
    REQUIRE(p.value() == 6.0);
    REQUIRE(p.grad()(0) == 3.0);
    REQUIRE(p.grad()(1) == 2.0);

    Dual6 q = x / y; // d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
    REQUIRE_THAT(q.value(), Catch::Matchers::WithinAbs(2.0 / 3.0, 1e-12));
    REQUIRE_THAT(q.grad()(0), Catch::Matchers::WithinAbs(1.0 / 3.0, 1e-12));
    REQUIRE_THAT(q.grad()(1), Catch::Matchers::WithinAbs(-2.0 / 9.0, 1e-12));

    Dual6 r = sqrt(x * x + y * y); // norm; d/dx = x/r, d/dy = y/r
    const double rv = std::sqrt(13.0);
    REQUIRE_THAT(r.value(), Catch::Matchers::WithinAbs(rv, 1e-12));
    REQUIRE_THAT(r.grad()(0), Catch::Matchers::WithinAbs(2.0 / rv, 1e-10));
    REQUIRE_THAT(r.grad()(1), Catch::Matchers::WithinAbs(3.0 / rv, 1e-10));
}

TEST_CASE("se3::Exp(0) is identity", "[se3]") {
    Matrix4d g = se3::Exp(Vector6d::Zero());
    REQUIRE(g.isApprox(Matrix4d::Identity(), 1e-14));
}

TEST_CASE("se3::Exp produces orthonormal rotation across scales", "[se3]") {
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (double scale : {1e-10, 1e-6, 1e-3, 1.0, 3.0}) {
        Vector6d xi;
        for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
        xi *= scale;
        Matrix4d g = se3::Exp(xi);
        REQUIRE(isOrthonormal(g.block<3, 3>(0, 0), 1e-9));
    }
}

TEST_CASE("se3::SE3Inverse round-trip", "[se3]") {
    std::mt19937 rng(7);
    for (int t = 0; t < 20; ++t) {
        Matrix4d g = randomPose(rng);
        Matrix4d ginv = se3::SE3Inverse(g);
        Matrix4d I1 = g * ginv;
        Matrix4d I2 = ginv * g;
        REQUIRE(I1.isApprox(Matrix4d::Identity(), 1e-9));
        REQUIRE(I2.isApprox(Matrix4d::Identity(), 1e-9));
    }
}

TEST_CASE("se3::relative matches g1^{-1} g2", "[se3]") {
    std::mt19937 rng(11);
    Matrix4d g1 = randomPose(rng);
    Matrix4d g2 = randomPose(rng);
    Matrix4d rel = se3::relative(g1, g2);
    Matrix4d expected = se3::SE3Inverse(g1) * g2;
    REQUIRE(rel.isApprox(expected, 1e-12));
}

// --- The core check: retract<Dual6>'s Jacobian vs. finite difference on ---
// --- the exact exponential map.                                        ---
TEST_CASE("se3::retract<Dual6> derivative matches finite difference on Exp", "[se3][dual6]") {
    std::mt19937 rng(123);

    // A handful of scalar probe functions of g exercising +,-,*,rotation and
    // translation blocks together, the way problem_matrices/kkt_R will.
    auto probes = std::vector<std::function<double(const Matrix4d&)>>{
        [](const Matrix4d& g) { return g(0, 3); },
        [](const Matrix4d& g) { return g.block<3, 3>(0, 0).trace(); },
        [](const Matrix4d& g) { return g(0, 3) * g(1, 0) + g(2, 3); },
        [](const Matrix4d& g) {
            Eigen::Vector3d a(0.3, -0.7, 1.1);
            return (g.block<3, 3>(0, 0) * a + g.block<3, 1>(0, 3)).squaredNorm();
        },
    };

    auto probeDual = [](const Eigen::Matrix<Dual6, 4, 4>& g, int which) -> Dual6 {
        switch (which) {
            case 0: return g(0, 3);
            case 1: return g(0, 0) + g(1, 1) + g(2, 2);
            case 2: return g(0, 3) * g(1, 0) + g(2, 3);
            default: {
                Eigen::Matrix<Dual6, 3, 1> a;
                a << Dual6(0.3), Dual6(-0.7), Dual6(1.1);
                Eigen::Matrix<Dual6, 3, 1> y = g.block<3, 3>(0, 0) * a + g.block<3, 1>(0, 3);
                return y.dot(y);
            }
        }
    };

    const double eps = 1e-6;
    for (int t = 0; t < 10; ++t) {
        Matrix4d g0 = randomPose(rng);

        for (std::size_t which = 0; which < probes.size(); ++which) {
            // Analytic via Dual6 retraction.
            Eigen::Matrix<Dual6, 4, 4> gD = se3::retract<Dual6>(g0, se3::seedTwist());
            Dual6 fD = probeDual(gD, static_cast<int>(which));
            Eigen::Matrix<double, 6, 1> analytic = fD.grad();

            // Finite difference via the exact exponential map.
            Eigen::Matrix<double, 6, 1> fd;
            for (int i = 0; i < 6; ++i) {
                Vector6d ei = Vector6d::Zero();
                ei(i) = eps;
                Matrix4d gp = g0 * se3::Exp(ei);
                Matrix4d gm = g0 * se3::Exp(-ei);
                fd(i) = (probes[which](gp) - probes[which](gm)) / (2.0 * eps);
            }

            INFO("trial " << t << " probe " << which);
            REQUIRE((analytic - fd).norm() < 1e-5);
        }
    }
}

// --- tangent_se3: the exact (closed-form, non-autodiff) Jacobian of Exp ---
// --- at a general point, validated against central-FD of Exp itself.   ---
TEST_CASE("se3::tangent_se3 matches finite difference of Exp", "[se3]") {
    std::mt19937 rng(77);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Vector6d xi;
        for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
        xi *= 0.8; // stay away from the wrap-around edge of the rotation part

        const Eigen::Matrix<double, 6, 6> T_analytic = se3::tangent_se3(xi);

        // Left-trivialized (empirically confirmed, not assumed): Exp(xi +
        // dxi) ~= Exp(T(xi)*dxi) * Exp(xi), i.e. T(xi) maps a small
        // coordinate change dxi to the LEFT/world-frame local twist that
        // reproduces it, to O(dxi^2).
        const Matrix4d g_base = se3::Exp(xi);
        Vector6d dxi;
        for (int i = 0; i < 6; ++i) dxi(i) = eps * nd(rng);
        const Matrix4d g_perturbed = se3::Exp(xi + dxi);
        const Matrix4d g_predicted = se3::Exp(T_analytic * dxi) * g_base;

        INFO("trial " << t);
        REQUIRE(g_perturbed.isApprox(g_predicted, 1e-8));
    }
}

// --- tangentDot_se3: directional derivative of tangent_se3, validated ---
// --- against central-FD of tangent_se3 itself along the given direction.
TEST_CASE("se3::tangentDot_se3 matches finite difference of tangent_se3", "[se3]") {
    std::mt19937 rng(88);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Vector6d xi, xi_dot;
        for (int i = 0; i < 6; ++i) {
            xi(i) = nd(rng);
            xi_dot(i) = nd(rng);
        }
        xi *= 0.8;

        const Eigen::Matrix<double, 6, 6> Tdot_analytic = se3::tangentDot_se3(xi, xi_dot);
        const Eigen::Matrix<double, 6, 6> Tp = se3::tangent_se3(xi + eps * xi_dot);
        const Eigen::Matrix<double, 6, 6> Tm = se3::tangent_se3(xi - eps * xi_dot);
        const Eigen::Matrix<double, 6, 6> Tdot_fd = (Tp - Tm) / (2.0 * eps);

        INFO("trial " << t);
        REQUIRE((Tdot_analytic - Tdot_fd).norm() < 1e-5);
    }
}

// --- tangentRight: the convention DCOL++ actually uses everywhere (right ---
// --- multiplication, local-frame twist -- matches `retract`).            --
TEST_CASE("se3::tangentRight matches finite difference of Exp (right-multiplication convention)", "[se3]") {
    std::mt19937 rng(66);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Vector6d xi;
        for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
        xi *= 0.8;

        const Eigen::Matrix<double, 6, 6> T_analytic = se3::tangentRight(xi);

        // Exp(xi+dxi) ~= Exp(xi) * Exp(tangentRight(xi)*dxi), O(dxi^2).
        const Matrix4d g_base = se3::Exp(xi);
        Vector6d dxi;
        for (int i = 0; i < 6; ++i) dxi(i) = eps * nd(rng);
        const Matrix4d g_perturbed = se3::Exp(xi + dxi);
        const Matrix4d g_predicted = g_base * se3::Exp(T_analytic * dxi);

        INFO("trial " << t);
        REQUIRE(g_perturbed.isApprox(g_predicted, 1e-8));
    }
}

TEST_CASE("se3::tangentDotRight matches finite difference of tangentRight", "[se3]") {
    std::mt19937 rng(55);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Vector6d xi, xi_dot;
        for (int i = 0; i < 6; ++i) {
            xi(i) = nd(rng);
            xi_dot(i) = nd(rng);
        }
        xi *= 0.8;

        const Eigen::Matrix<double, 6, 6> Tdot_analytic = se3::tangentDotRight(xi, xi_dot);
        const Eigen::Matrix<double, 6, 6> Tp = se3::tangentRight(xi + eps * xi_dot);
        const Eigen::Matrix<double, 6, 6> Tm = se3::tangentRight(xi - eps * xi_dot);
        const Eigen::Matrix<double, 6, 6> Tdot_fd = (Tp - Tm) / (2.0 * eps);

        INFO("trial " << t);
        REQUIRE((Tdot_analytic - Tdot_fd).norm() < 1e-5);
    }
}
