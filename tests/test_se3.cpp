// dcolpp::se3 -- Exp and the pose-derivative primitives (dPointDXi family),
// each checked against central finite differences of the exact
// (non-linearized) quantity it claims to differentiate.

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
