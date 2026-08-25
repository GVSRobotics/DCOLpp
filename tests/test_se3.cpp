// dcolpp::se3 -- Exp, its Jacobians (tangent_se3 family), and the
// pose-derivative primitives (dPointDXi family), each checked against
// central finite differences of the exact (non-linearized) quantity it
// claims to differentiate.

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>
#include "portable_random.hpp"

#include "dcolpp/se3.hpp"

using namespace dcolpp;
using Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomPose(std::mt19937& rng) {
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.head<3>() *= 0.8; // stay away from the rotation part's wrap-around edge
    Matrix4d g = se3::Exp(xi);
    g.block<3, 1>(0, 3) += Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

bool isOrthonormal(const Eigen::Matrix3d& R, double tol = 1e-9) {
    return (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < tol;
}

} // namespace

TEST_CASE("se3::Exp(0) is identity", "[se3]") {
    Matrix4d g = se3::Exp(Vector6d::Zero());
    REQUIRE(g.isApprox(Matrix4d::Identity(), 1e-14));
}

TEST_CASE("se3::Exp produces orthonormal rotation across scales", "[se3]") {
    std::mt19937 rng(42);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    for (double scale : {1e-10, 1e-6, 1e-3, 1.0, 3.0}) {
        Vector6d xi;
        for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
        xi *= scale;
        Matrix4d g = se3::Exp(xi);
        REQUIRE(isOrthonormal(g.block<3, 3>(0, 0), 1e-9));
    }
}

TEST_CASE("se3::retract matches g0*Exp(xi)", "[se3]") {
    std::mt19937 rng(21);
    Matrix4d g0 = randomPose(rng);
    Vector6d xi = Vector6d::Random() * 0.5;
    REQUIRE(se3::retract<double>(g0, xi).isApprox(g0 * se3::Exp(xi), 1e-14));
}

TEST_CASE("se3::SE3Inverse round-trip", "[se3]") {
    std::mt19937 rng(7);
    for (int t = 0; t < 20; ++t) {
        Matrix4d g = randomPose(rng);
        Matrix4d ginv = se3::SE3Inverse(g);
        REQUIRE((g * ginv).isApprox(Matrix4d::Identity(), 1e-9));
        REQUIRE((ginv * g).isApprox(Matrix4d::Identity(), 1e-9));
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

// tangent_se3: the exact (closed-form) Jacobian of Exp at a general point.
// Left-trivialized (confirmed, not assumed): Exp(xi+dxi) ~= Exp(T(xi)*dxi) *
// Exp(xi), i.e. T(xi) maps a coordinate change dxi to the left/world-frame
// local twist that reproduces it, to O(dxi^2).
TEST_CASE("se3::tangent_se3 matches finite difference of Exp", "[se3]") {
    std::mt19937 rng(77);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Vector6d xi;
        for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
        xi *= 0.8;

        const Eigen::Matrix<double, 6, 6> T_analytic = se3::tangent_se3(xi);
        const Matrix4d g_base = se3::Exp(xi);
        Vector6d dxi;
        for (int i = 0; i < 6; ++i) dxi(i) = eps * nd(rng);
        const Matrix4d g_perturbed = se3::Exp(xi + dxi);
        const Matrix4d g_predicted = se3::Exp(T_analytic * dxi) * g_base;

        INFO("trial " << t);
        REQUIRE(g_perturbed.isApprox(g_predicted, 1e-8));
    }
}

// tangentDot_se3: directional derivative of tangent_se3, vs. central-FD of
// tangent_se3 itself along the given direction.
TEST_CASE("se3::tangentDot_se3 matches finite difference of tangent_se3", "[se3]") {
    std::mt19937 rng(88);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
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

// tangentRight: the convention DCOL++ actually uses (right multiplication,
// local-frame twist -- matches retract).
TEST_CASE("se3::tangentRight matches finite difference of Exp (right-multiplication convention)", "[se3]") {
    std::mt19937 rng(66);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
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
    dcolpp_test::PortableNormal nd(0.0, 1.0);
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

TEST_CASE("se3::dPointDXi matches finite difference at xi=0", "[se3]") {
    std::mt19937 rng(101);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Matrix4d g0 = randomPose(rng);
        Eigen::Vector3d r_local(nd(rng), nd(rng), nd(rng));

        const Eigen::Matrix<double, 3, 6> analytic = se3::dPointDXi(g0, r_local);

        Eigen::Matrix<double, 3, 6> fd;
        for (int i = 0; i < 6; ++i) {
            Vector6d ei = Vector6d::Zero();
            ei(i) = eps;
            const Matrix4d gp = g0 * se3::Exp(ei);
            const Matrix4d gm = g0 * se3::Exp(-ei);
            const Eigen::Vector3d pp = gp.block<3, 3>(0, 0) * r_local + gp.block<3, 1>(0, 3);
            const Eigen::Vector3d pm = gm.block<3, 3>(0, 0) * r_local + gm.block<3, 1>(0, 3);
            fd.col(i) = (pp - pm) / (2.0 * eps);
        }

        INFO("trial " << t);
        REQUIRE((analytic - fd).norm() < 1e-6);
    }
}

TEST_CASE("se3::dInverseRotatedVectorDXi matches finite difference at xi=0", "[se3]") {
    std::mt19937 rng(103);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Matrix4d g0 = randomPose(rng);
        Eigen::Vector3d w(nd(rng), nd(rng), nd(rng));

        const Eigen::Matrix<double, 3, 6> analytic = se3::dInverseRotatedVectorDXi(g0, w);

        Eigen::Matrix<double, 3, 6> fd;
        for (int i = 0; i < 6; ++i) {
            Vector6d ei = Vector6d::Zero();
            ei(i) = eps;
            const Matrix4d gp = g0 * se3::Exp(ei);
            const Matrix4d gm = g0 * se3::Exp(-ei);
            const Eigen::Vector3d vp = gp.block<3, 3>(0, 0).transpose() * w;
            const Eigen::Vector3d vm = gm.block<3, 3>(0, 0).transpose() * w;
            fd.col(i) = (vp - vm) / (2.0 * eps);
        }

        INFO("trial " << t);
        REQUIRE((analytic - fd).norm() < 1e-6);
    }
}

TEST_CASE("se3::dRotatedVectorDXi matches finite difference at xi=0", "[se3]") {
    std::mt19937 rng(102);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    const double eps = 1e-6;

    for (int t = 0; t < 10; ++t) {
        Matrix4d g0 = randomPose(rng);
        Eigen::Vector3d v_local(nd(rng), nd(rng), nd(rng));

        const Eigen::Matrix<double, 3, 6> analytic = se3::dRotatedVectorDXi(g0, v_local);

        Eigen::Matrix<double, 3, 6> fd;
        for (int i = 0; i < 6; ++i) {
            Vector6d ei = Vector6d::Zero();
            ei(i) = eps;
            const Matrix4d gp = g0 * se3::Exp(ei);
            const Matrix4d gm = g0 * se3::Exp(-ei);
            const Eigen::Vector3d vp = gp.block<3, 3>(0, 0) * v_local;
            const Eigen::Vector3d vm = gm.block<3, 3>(0, 0) * v_local;
            fd.col(i) = (vp - vm) / (2.0 * eps);
        }

        INFO("trial " << t);
        REQUIRE((analytic - fd).norm() < 1e-6);
    }
}
