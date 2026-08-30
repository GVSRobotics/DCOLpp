#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

#include "dcolpp/socp/proximity.hpp"
#include "socp_ref_case.hpp"
#include "generated/socp_reference_data.hpp"

using namespace dcolpp::socp;
using dcolpp::test_ref::SocpRefCase;
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace {

constexpr double kPi = 3.14159265358979323846;

Matrix4d loadG(const SocpRefCase& c) {
    Matrix4d g;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) g(i, j) = c.g[i * 4 + j]; // row-major source
    return g;
}

template <typename Shape1, typename Shape2>
void checkAgainstJulia(const Shape1& s1, const Shape2& s2, const SocpRefCase* cases, int n,
                        const char* label) {
    // Generic, not the library default (Geometric, SocpOptions::init_strategy):
    // this test validates exact numerical agreement with
    // DifferentiableCollisions.jl's own reference output, which was
    // produced by (and only ever validated against) the original
    // least-squares-then-bring2cone initialization DCOL++ started with
    // (DEVIATIONS.md §1, "Unchanged from Julia"). The geometric init
    // (DEVIATIONS.md "geometric initial guess") is a genuinely different,
    // independently-derived starting point -- correct, faster on average,
    // but not guaranteed to retrace Julia's exact solving trajectory
    // through a non-strictly-convex SOCP (this pair's objective is linear;
    // a polygon-edge ORT constraint being exactly active on one reference
    // case makes the witness point's own optimal set locally near-flat, so
    // different KKT-valid paths can legitimately land at different points
    // within it -- verified directly, not assumed). Generic keeps this
    // test doing what it's actually for: checking the port's fidelity to
    // its source, not comparing two independently-valid solving strategies.
    SocpOptions opt;
    opt.pdip_tol = 1e-10;
    opt.init_strategy = SocpInitStrategy::Generic;
    for (int i = 0; i < n; ++i) {
        const SocpRefCase& c = cases[i];
        Matrix4d g = loadG(c);
        ProximityResult res = proximity(s1, s2, g, opt);

        INFO(label << " case " << i);
        REQUIRE(res.converged);
        REQUIRE(std::abs(res.alpha - c.alpha) < 1e-6);
        Vector3d expected(c.witness[0], c.witness[1], c.witness[2]);
        REQUIRE((res.witness_point - expected).norm() < 1e-5);
    }
}

Polytope<6> makeCube() {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    return Polytope<6>(A, b);
}

Polygon<6> makeHex() {
    Eigen::Matrix<double, 6, 2> A;
    Eigen::Matrix<double, 6, 1> b;
    for (int i = 0; i < 6; ++i) {
        const double th = 2.0 * kPi * i / 6.0;
        A(i, 0) = std::cos(th);
        A(i, 1) = std::sin(th);
    }
    b.setConstant(0.4);
    return Polygon<6>(A, b, 0.1);
}

Ellipsoid makeEllipsoid() { return Ellipsoid(0.6, 0.4, 0.5); }

} // namespace

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: sphere-sphere", "[socp][julia-parity]") {
    checkAgainstJulia(Sphere(0.5), Sphere(0.3), dcolpp::test_ref::SphereSphere,
                       dcolpp::test_ref::SphereSphereCount, "SphereSphere");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: sphere-capsule", "[socp][julia-parity]") {
    checkAgainstJulia(Sphere(0.4), Capsule(0.3, 1.2), dcolpp::test_ref::SphereCapsule,
                       dcolpp::test_ref::SphereCapsuleCount, "SphereCapsule");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: capsule-cylinder", "[socp][julia-parity]") {
    checkAgainstJulia(Capsule(0.3, 1.2), Cylinder(0.25, 0.9), dcolpp::test_ref::CapsuleCylinder,
                       dcolpp::test_ref::CapsuleCylinderCount, "CapsuleCylinder");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: cylinder-cone", "[socp][julia-parity]") {
    checkAgainstJulia(Cylinder(0.25, 0.9), Cone(1.2, 22.0 * kPi / 180.0), dcolpp::test_ref::CylinderCone,
                       dcolpp::test_ref::CylinderConeCount, "CylinderCone");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: cone-polytope", "[socp][julia-parity]") {
    checkAgainstJulia(Cone(1.2, 22.0 * kPi / 180.0), makeCube(), dcolpp::test_ref::ConePolytope,
                       dcolpp::test_ref::ConePolytopeCount, "ConePolytope");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: polytope-ellipsoid", "[socp][julia-parity]") {
    checkAgainstJulia(makeCube(), makeEllipsoid(), dcolpp::test_ref::PolytopeEllipsoid,
                       dcolpp::test_ref::PolytopeEllipsoidCount, "PolytopeEllipsoid");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: ellipsoid-polygon", "[socp][julia-parity]") {
    checkAgainstJulia(makeEllipsoid(), makeHex(), dcolpp::test_ref::EllipsoidPolygon,
                       dcolpp::test_ref::EllipsoidPolygonCount, "EllipsoidPolygon");
}

TEST_CASE("socp forward solve matches DifferentiableCollisions.jl: polygon-sphere", "[socp][julia-parity]") {
    checkAgainstJulia(makeHex(), Sphere(0.35), dcolpp::test_ref::PolygonSphere,
                       dcolpp::test_ref::PolygonSphereCount, "PolygonSphere");
}
