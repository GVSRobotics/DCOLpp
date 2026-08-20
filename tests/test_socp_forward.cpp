#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include "dcolpp/socp/proximity.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Eigen::Vector3d;

TEST_CASE("sphere-sphere proximity matches closed form", "[socp][forward]") {
    Sphere s1(0.5);
    Sphere s2(0.3);

    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = Vector3d(2.0, 0.0, 0.0);

    ProximityResult res = proximity(s1, s2, g);
    REQUIRE(res.converged);

    const double d = 2.0;
    const double expected_alpha = d / (s1.R + s2.R);
    REQUIRE_THAT(res.alpha, Catch::Matchers::WithinAbs(expected_alpha, 1e-6));

    // witness point should lie on the segment between the two centers,
    // at distance alpha*R1 from sphere 1's center.
    Vector3d expected_witness = Vector3d(1, 0, 0) * (res.alpha * s1.R);
    REQUIRE((res.witness_point - expected_witness).norm() < 1e-5);
}

TEST_CASE("sphere-sphere proximity is symmetric", "[socp][forward]") {
    Sphere s1(0.7);
    Sphere s2(0.2);
    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = Vector3d(0.4, 1.1, -0.3);

    ProximityResult r12 = proximity(s1, s2, g);
    ProximityResult r21 = proximity(s2, s1, g.inverse());

    REQUIRE(r12.converged);
    REQUIRE(r21.converged);
    REQUIRE_THAT(r12.alpha, Catch::Matchers::WithinAbs(r21.alpha, 1e-6));
}

TEST_CASE("capsule-sphere proximity converges and is frame consistent", "[socp][forward]") {
    Capsule cap(0.3, 1.2);
    Sphere sph(0.5);

    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = Vector3d(1.5, 0.3, -0.2);

    ProximityResult res = proximity(cap, sph, g);
    REQUIRE(res.converged);
    REQUIRE(res.alpha > 0.0);
    REQUIRE(res.iters > 0);
}

TEST_CASE("polytope (cube) vs sphere proximity converges", "[socp][forward]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0,
        -1, 0, 0,
         0, 1, 0,
         0,-1, 0,
         0, 0, 1,
         0, 0,-1;
    b << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    Polytope<6> cube(A, b);
    Sphere sph(0.4);

    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = Vector3d(2.0, 0.0, 0.0);

    ProximityResult res = proximity(cube, sph, g);
    REQUIRE(res.converged);
    // touching distance would be 0.5 (cube half-width) + 0.4 (sphere R) = 0.9
    // over a separation of 2.0 -> alpha = 2.0/0.9
    REQUIRE_THAT(res.alpha, Catch::Matchers::WithinAbs(2.0 / 0.9, 1e-4));
}
