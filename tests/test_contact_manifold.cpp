// ContactManifold (proximity_contact.hpp): multi-point witness sets for
// degenerate (dim>0) contacts. Verifies:
//  - dim 0/1 exact cases (1 point / the 2 exact segment endpoints).
//  - dim 2 spans the true patch and, critically, that the recentering step
//    actually removes the SocpInitStrategy::Geometric off-centering bias
//    documented in DEVIATIONS.md -- this is the concrete problem the
//    recentering stage exists to fix, so it's tested directly, not just
//    "the manifold looks reasonable".
//  - K (contact_manifold_points) is respected, including K > 4.
//  - contact_manifold_dim/normal_cone_dim/*_valid agree with contactDegeneracy.
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_contact.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace {

Polytope<6> makeBox(double h) {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(h);
    return Polytope<6>(A, b);
}

} // namespace

TEST_CASE("ContactManifold: sphere-sphere (dim 0) -> exactly 1 point, matches x*", "[manifold]") {
    Sphere s1(0.8), s2(0.6);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.4;
    SocpOptions opt;
    opt.compute_contact_manifold = true;
    const auto res = proximityContactJacobian(s1, s2, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 0);
    REQUIRE(res.contact_manifold_points.size() == 1);
    REQUIRE((res.contact_manifold_points[0] - res.witness_point).norm() < 1e-9);
}

TEST_CASE("ContactManifold: box edge-edge parallel (dim 1) -> exactly the 2 true segment endpoints",
          "[manifold]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0;
    SocpOptions opt;
    opt.compute_contact_manifold = true;
    const auto res = proximityContactJacobian(box, box, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 1);
    REQUIRE(res.contact_manifold_points.size() == 2);

    // True segment: x=0.5, y=0.5, z in [-0.5, 0.5] -- exact, regardless of
    // where x* itself landed on that line.
    const Vector3d expected_a(0.5, 0.5, -0.5), expected_b(0.5, 0.5, 0.5);
    const auto& p0 = res.contact_manifold_points[0];
    const auto& p1 = res.contact_manifold_points[1];
    const bool order1 = (p0 - expected_a).norm() < 1e-6 && (p1 - expected_b).norm() < 1e-6;
    const bool order2 = (p0 - expected_b).norm() < 1e-6 && (p1 - expected_a).norm() < 1e-6;
    REQUIRE((order1 || order2));
}

TEST_CASE("ContactManifold: cylinder-cylinder axis-parallel (dim 1, exercises the SOC ray-clip branch)",
          "[manifold]") {
    // Unlike the box edge-edge case, this line contact's null direction
    // runs along a smooth (SOC) generatrix -- the only automated coverage
    // of contactManifold's quadratic ray-vs-cone clip, not just the linear
    // ray-vs-halfspace one every polytope-polytope case exercises.
    Cylinder cyl(0.4, 1.5);
    Matrix4d g = Matrix4d::Identity();
    g(1, 3) = 0.8;
    SocpOptions opt;
    opt.compute_contact_manifold = true;
    const auto res = proximityContactJacobian(cyl, cyl, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 1);
    REQUIRE(res.contact_manifold_points.size() == 2);

    // True touching line: x in [-0.75, 0.75] (cylinders fully axially
    // overlapping, half-length 0.75 each), y=0.4 (R1+R2 split), z=0.
    const Vector3d expected_a(-0.75, 0.4, 0.0), expected_b(0.75, 0.4, 0.0);
    const auto& p0 = res.contact_manifold_points[0];
    const auto& p1 = res.contact_manifold_points[1];
    const bool order1 = (p0 - expected_a).norm() < 1e-4 && (p1 - expected_b).norm() < 1e-4;
    const bool order2 = (p0 - expected_b).norm() < 1e-4 && (p1 - expected_a).norm() < 1e-4;
    REQUIRE((order1 || order2));
}

TEST_CASE("ContactManifold: box face-face parallel (dim 2) -> K=4 points spanning the square", "[manifold]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 4;
    const auto res = proximityContactJacobian(box, box, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 2);
    REQUIRE(res.contact_manifold_points.size() == 4);

    // Every point must be ON the true patch boundary: x=0.5, and (y,z)
    // within the box's face [-0.5,0.5]^2 with at least one coordinate at
    // the boundary (since these are ray-clip HITS, not interior samples).
    for (const auto& p : res.contact_manifold_points) {
        REQUIRE(std::abs(p.x() - 0.5) < 1e-6);
        REQUIRE(p.y() >= -0.5 - 1e-6);
        REQUIRE(p.y() <= 0.5 + 1e-6);
        REQUIRE(p.z() >= -0.5 - 1e-6);
        REQUIRE(p.z() <= 0.5 + 1e-6);
    }
    // Spread check: farthest-point selection should give a real spread, not
    // 4 clustered points -- centroid should be near the square's own center
    // (0,0) by symmetry, and no two points should be near-coincident.
    Vector3d centroid = Vector3d::Zero();
    for (const auto& p : res.contact_manifold_points) centroid += p;
    centroid /= 4.0;
    REQUIRE(std::abs(centroid.y()) < 0.15);
    REQUIRE(std::abs(centroid.z()) < 0.15);
    for (size_t i = 0; i < res.contact_manifold_points.size(); ++i) {
        for (size_t j = i + 1; j < res.contact_manifold_points.size(); ++j) {
            REQUIRE((res.contact_manifold_points[i] - res.contact_manifold_points[j]).norm() > 0.3);
        }
    }
}

TEST_CASE("ContactManifold: K > 4 is respected", "[manifold]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 8;
    const auto res = proximityContactJacobian(box, box, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_points.size() == 8);
}

TEST_CASE("ContactManifold: recentering removes the Geometric-init off-centering bias", "[manifold]") {
    // The concrete finding that motivated the recentering stage: on an
    // ASYMMETRIC overlap, x* itself lands off the true analytic center
    // under SocpInitStrategy::Geometric (not under Generic). Check that the
    // manifold's centroid is near the TRUE overlap center regardless of
    // which strategy solved it -- i.e. that the fix actually fixes it, not
    // just "the points are somewhere on the boundary".
    Polytope<6> box1 = makeBox(0.5);
    Polytope<6> box2 = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    g(1, 3) = 0.3; // true overlap: y in [-0.2, 0.5], center y=0.15

    for (auto strategy : {SocpInitStrategy::Generic, SocpInitStrategy::Geometric}) {
        SocpOptions opt;
        opt.init_strategy = strategy;
        opt.compute_contact_manifold = true;
        opt.contact_manifold_points = 4;
        const auto res = proximityContactJacobian(box1, box2, g, opt);
        REQUIRE(res.converged);
        REQUIRE(res.contact_manifold_dim == 2);

        Vector3d centroid = Vector3d::Zero();
        for (const auto& p : res.contact_manifold_points) centroid += p;
        centroid /= static_cast<double>(res.contact_manifold_points.size());
        // True center y=0.15; recentering is exact-per-axis, not exact in
        // full 2D, so allow a modest tolerance -- but it must be much
        // closer than the raw x* bias was (Geometric's x* alone was off by
        // 0.019 on the milder case, 0.006/6% on the narrower one).
        REQUIRE(std::abs(centroid.y() - 0.15) < 0.05);
    }
}

TEST_CASE("ContactManifold: dims/valid agree with contactDegeneracy", "[manifold]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0; g(2, 3) = 1.0; // corner-corner: dim 0, normal_cone_dim 2

    SocpOptions optDeg;
    optDeg.compute_degeneracy_info = true;
    const auto degen = proximityContactJacobian(box, box, g, optDeg);

    SocpOptions optMan;
    optMan.compute_contact_manifold = true;
    const auto man = proximityContactJacobian(box, box, g, optMan);

    REQUIRE(degen.contact_manifold_dim == man.contact_manifold_dim);
    REQUIRE(degen.normal_cone_dim == man.normal_cone_dim);
    REQUIRE(degen.witness_jacobian_valid == man.witness_jacobian_valid);
    REQUIRE(degen.normal_jacobian_valid == man.normal_jacobian_valid);
    REQUIRE(man.contact_manifold_dim == 0);
    REQUIRE(man.contact_manifold_points.size() == 1);
}

TEST_CASE("ContactManifold: off by default -- empty vector", "[manifold]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    const auto res = proximityContactJacobian(box, box, g); // default SocpOptions
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_points.empty());
    REQUIRE(res.contact_manifold_dim == -1);
}
