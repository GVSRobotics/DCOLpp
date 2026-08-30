// proximityContact / proximityContactJacobian (proximity_contact.hpp): the
// alpha+witness_point+normal bundle, with and without Jacobians wrt xi.
//
// The underlying math (computeProximityGradient, diffSocp,
// computeContactNormalJacobian) is already verified against FD elsewhere
// (test_analytic_derivatives.cpp, test_hessian_derivatives.cpp). What these
// tests actually exercise is the wiring in proximity_contact.hpp itself --
// template parameter bookkeeping, which (x,s,z) get passed where -- by
// cross-checking against independent direct calls to those already-verified
// functions.

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>
#include "portable_random.hpp"

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_contact.hpp"
#include "dcolpp/socp/proximity_gradient.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomG(std::mt19937& rng, double translation_scale = 2.5) {
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.head<3>() *= 0.8;
    Matrix4d g = dcolpp::se3::Exp(xi);
    g.block<3, 1>(0, 3) += translation_scale * Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

template <typename Shape1, typename Shape2>
void checkProximityContact(const Shape1& s1, const Shape2& s2, int n_poses, unsigned seed) {
    std::mt19937 rng(seed);
    for (int i = 0; i < n_poses; ++i) {
        const Matrix4d g = randomG(rng);

        const auto res = proximityContact(s1, s2, g);
        REQUIRE(res.converged);

        // Independent ground truth: a separate proximityGradient() solve +
        // contactNormal(), exactly as a caller would have composed these by
        // hand before this helper existed.
        const auto gr = proximityGradient(s1, s2, g);
        REQUIRE(gr.converged);
        const Vector3d expected_normal = contactNormal(gr, g);

        REQUIRE(std::abs(res.alpha - gr.alpha) < 1e-9);
        REQUIRE((res.witness_point - gr.witness_point).norm() < 1e-9);
        REQUIRE((res.normal - expected_normal).norm() < 1e-9);
        REQUIRE(std::abs(res.normal.norm() - 1.0) < 1e-9);
    }
}

template <typename Shape1, typename Shape2>
void checkProximityContactJacobian(const Shape1& s1, const Shape2& s2, int n_poses, unsigned seed) {
    std::mt19937 rng(seed);
    for (int i = 0; i < n_poses; ++i) {
        const Matrix4d g = randomG(rng);

        const auto res = proximityContactJacobian(s1, s2, g);
        REQUIRE(res.converged);

        // Independent ground truth: build the SOCP directly, solve it, and
        // call the already-FD-verified analytic functions by hand with
        // explicit template parameters (as test_hessian_derivatives.cpp
        // does), rather than going through proximity_contact.hpp's wrappers.
        const Matrix4d I4 = Matrix4d::Identity();
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        const auto sol = solveProximitySocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            s1, s2, g, combined.c, combined.G, combined.h, SocpOptions{});
        REQUIRE(sol.converged);

        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

        const Eigen::Matrix<double, 4, 6> expected_jacobian =
            diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
                s1, s2, sol.x, sol.s, sol.z, g, combined.G);

        const Eigen::Matrix<double, 1, 6> grad =
            computeProximityGradient<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                s1, s2, sol.x, sol.z, g);
        const Vector3d expected_normal = (g.block<3, 3>(0, 0) * grad.template tail<3>().transpose()).normalized();

        const Eigen::Matrix<double, 3, 6> expected_normal_jacobian =
            computeContactNormalJacobian<Shape1, Shape2, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                s1, s2, sol.x, sol.s, sol.z, g);

        REQUIRE(std::abs(res.alpha - sol.x(3)) < 1e-12);
        REQUIRE((res.witness_point - sol.x.template head<3>()).norm() < 1e-12);
        REQUIRE((res.normal - expected_normal).norm() < 1e-12);
        REQUIRE((res.jacobian - expected_jacobian).norm() < 1e-12);
        REQUIRE((res.normal_jacobian - expected_normal_jacobian).norm() < 1e-12);
    }
}

Polygon<4> unitSquarePolygon(double R) {
    Eigen::Matrix<double, 4, 2> A;
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    Eigen::Matrix<double, 4, 1> b = Eigen::Matrix<double, 4, 1>::Constant(1.0);
    return Polygon<4>(A, b, R);
}

} // namespace

TEST_CASE("proximityContact matches proximityGradient+contactNormal: Sphere vs Sphere", "[contact]") {
    checkProximityContact(Sphere(1.3), Sphere(0.7), 15, 700);
}

TEST_CASE("proximityContact matches proximityGradient+contactNormal: Capsule vs Cone", "[contact]") {
    checkProximityContact(Capsule(0.5, 2.0), Cone(1.5, 0.35), 15, 701);
}

TEST_CASE("proximityContact matches proximityGradient+contactNormal: Polytope vs Ellipsoid", "[contact]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(0.6);
    checkProximityContact(Polytope<6>(A, b), Ellipsoid(0.9, 1.1, 1.4), 15, 702);
}

TEST_CASE("proximityContact matches proximityGradient+contactNormal: Polygon vs Cylinder", "[contact]") {
    checkProximityContact(unitSquarePolygon(0.2), Cylinder(0.6, 1.8), 15, 703);
}

TEST_CASE("proximityContact matches proximityGradient+contactNormal: Sphere vs TruncatedCone", "[contact]") {
    checkProximityContact(Sphere(0.8), TruncatedCone(0.9, 0.35, 1.6), 15, 704);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: Sphere vs Sphere", "[contact]") {
    checkProximityContactJacobian(Sphere(1.3), Sphere(0.7), 12, 710);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: Capsule vs Cone", "[contact]") {
    checkProximityContactJacobian(Capsule(0.5, 2.0), Cone(1.5, 0.35), 12, 711);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: Polytope vs Ellipsoid",
          "[contact]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(0.6);
    checkProximityContactJacobian(Polytope<6>(A, b), Ellipsoid(0.9, 1.1, 1.4), 12, 712);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: Polygon vs Cylinder",
          "[contact]") {
    checkProximityContactJacobian(unitSquarePolygon(0.2), Cylinder(0.6, 1.8), 12, 713);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: Sphere vs TruncatedCone",
          "[contact]") {
    checkProximityContactJacobian(Sphere(0.8), TruncatedCone(0.9, 0.35, 1.6), 12, 714);
}

TEST_CASE("proximityContactJacobian matches diffSocp+computeContactNormalJacobian: TruncatedCone vs Polytope",
          "[contact]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(0.6);
    checkProximityContactJacobian(TruncatedCone(0.9, 0.3, 1.5), Polytope<6>(A, b), 12, 715);
}
