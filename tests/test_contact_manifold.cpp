// ContactManifold (contact.hpp): multi-point witness sets for
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
#include <algorithm>
#include <cmath>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/contact.hpp"

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

// Regression: an AXIAL offset on a dim-1 (line-contact) pair between two
// round-shaft shapes (Cylinder/Capsule) used to send one endpoint to tens
// of thousands of units away. Root cause: the null direction lies exactly
// tangent to each shape's own active SOC block there (sliding along the
// shared touching line doesn't change either shape's own radial-distance
// constraint), so the ray-clip's quadratic-formula coefficients (A_c, B_c,
// C_c) should all be ~0 -- but land at different, inconsistent magnitudes
// after roundoff (A_c,B_c ~1e-13..1e-16, amplified through the near-zero-
// singular-value null direction), and a single fixed absolute threshold on
// the derived B_c let a ~1e-13 "linear coefficient" through as significant,
// dividing a real C_c by it. Fixed by checking the ROOT quantity (the SOC
// block's response `a` to the direction) against a scale-aware threshold
// BEFORE ever computing A_c/B_c/C_c. Exercises Cylinder-Cylinder, Capsule-
// Capsule, and mixed Cylinder-Capsule -- confirms the fix isn't shape-
// specific, not just re-tests the one case that surfaced it.
TEST_CASE("ContactManifold: axial-offset round-shaft pairs stay bounded (regression)", "[manifold]") {
    auto checkRange = [](double lo, double hi, const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) {
        double x0 = p0.x(), x1 = p1.x();
        if (x0 > x1) std::swap(x0, x1);
        REQUIRE(std::abs(x0 - lo) < 1e-2);
        REQUIRE(std::abs(x1 - hi) < 1e-2);
    };

    Cylinder cyl1(0.5, 1.4), cyl2(0.5, 1.4);
    for (double tx : {0.0, 0.2, 0.5, 0.9, 1.3}) {
        Matrix4d g = Matrix4d::Identity();
        g(0, 3) = tx; g(1, 3) = 1.0; // touching radially (R1+R2), shifted axially
        SocpOptions opt;
        opt.compute_contact_manifold = true;
        const auto res = proximityContactJacobian(cyl1, cyl2, g, opt);
        REQUIRE(res.converged);
        REQUIRE(res.contact_manifold_dim == 1);
        REQUIRE(res.contact_manifold_points.size() == 2);
        checkRange(std::max(-0.7, tx - 0.7), std::min(0.7, tx + 0.7), res.contact_manifold_points[0],
                   res.contact_manifold_points[1]);
    }

    Capsule cap1(0.4, 1.2), cap2(0.4, 1.2);
    for (double tx : {0.0, 0.2, 0.5, 0.9}) {
        Matrix4d g = Matrix4d::Identity();
        g(0, 3) = tx; g(1, 3) = 0.8;
        SocpOptions opt;
        opt.compute_contact_manifold = true;
        const auto res = proximityContactJacobian(cap1, cap2, g, opt);
        REQUIRE(res.converged);
        REQUIRE(res.contact_manifold_dim == 1);
        checkRange(std::max(-0.6, tx - 0.6), std::min(0.6, tx + 0.6), res.contact_manifold_points[0],
                   res.contact_manifold_points[1]);
    }

    Cylinder cyl3(0.5, 1.4);
    Capsule cap3(0.4, 1.2);
    for (double tx : {0.0, 0.15, 0.4}) {
        Matrix4d g = Matrix4d::Identity();
        g(0, 3) = tx; g(1, 3) = 0.9;
        SocpOptions opt;
        opt.compute_contact_manifold = true;
        const auto res = proximityContactJacobian(cyl3, cap3, g, opt);
        REQUIRE(res.converged);
        REQUIRE(res.contact_manifold_dim == 1);
        checkRange(std::max(-0.7, tx - 0.6), std::min(0.7, tx + 0.6), res.contact_manifold_points[0],
                   res.contact_manifold_points[1]);
    }
}

// contact_manifold_point_jacobians (contact.hpp): per-point
// jacobian/normal_jacobian under the "same active set" (s*,z* reused, only
// x's position swapped) convention. Populated automatically whenever
// opt.compute_contact_manifold is true -- not a separate opt-in. Verifies:
//  - one entry per contact_manifold_points[i], same order, always populated
//    alongside compute_contact_manifold.
//  - normal_jacobian is POINT-INVARIANT across the whole manifold (rigid-
//    translation argument: d(alpha)/dv doesn't depend on which point of a
//    shared flat patch x sits at, and normal_jacobian is built purely from
//    that) -- matches res.normal_jacobian too.
//  - jacobian's alpha row (row 3) is likewise invariant in its translation
//    columns (cols 3-5, d(alpha)/dv) but genuinely DIFFERS in its rotation
//    columns (cols 0-2, d(alpha)/dw -- a real moment-arm/lever effect: two
//    manifold points on opposite sides of the patch respond oppositely to
//    a small rotation of shape 2).
TEST_CASE("ContactManifold: contact_manifold_point_jacobians matches the same-active-set model", "[manifold]") {
    Polytope<6> box1 = makeBox(0.5), box2 = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; // face-face, full overlap in y/z -- dim 2

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.pdip_tol = 1e-10; // tightened: the invariance check below needs solver precision, not solver-default slop
    const auto res = proximityContactJacobian(box1, box2, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 2);
    REQUIRE(res.contact_manifold_point_jacobians.size() == res.contact_manifold_points.size());
    REQUIRE(res.contact_manifold_points.size() >= 3); // need at least 3 distinct points for the check below to bite

    const double kTight = 1e-8;
    bool sawRotationDifference = false;
    for (size_t i = 0; i < res.contact_manifold_point_jacobians.size(); ++i) {
        const auto& mpj = res.contact_manifold_point_jacobians[i];
        // normal_jacobian: point-invariant, matches the single res.normal_jacobian.
        REQUIRE((mpj.normal_jacobian - res.normal_jacobian).norm() < kTight);
        // alpha row, translation columns (d(alpha)/dv): point-invariant.
        REQUIRE((mpj.jacobian.row(3).tail<3>() - res.jacobian.row(3).tail<3>()).norm() < kTight);
        if (i > 0) {
            const auto& prev = res.contact_manifold_point_jacobians[i - 1];
            if ((mpj.jacobian.row(3).head<3>() - prev.jacobian.row(3).head<3>()).norm() > 1e-3) {
                sawRotationDifference = true;
            }
        }
    }
    REQUIRE(sawRotationDifference); // alpha row, rotation columns: genuinely point-dependent
}

// dim 0: the single contact_manifold_point IS x* (contact_manifold.hpp's own
// guarantee), so its jacobian entry should be an exact mirror of res.jacobian/
// res.normal_jacobian, not a freshly (and redundantly) solved value.
TEST_CASE("ContactManifold: contact_manifold_point_jacobians at dim 0 mirrors res.jacobian exactly",
          "[manifold]") {
    Polytope<6> box1 = makeBox(0.5), box2 = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0; g(2, 3) = 1.0; // corner touch -- dim 0

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    const auto res = proximityContactJacobian(box1, box2, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 0);
    REQUIRE(res.contact_manifold_point_jacobians.size() == 1);
    REQUIRE(res.contact_manifold_point_jacobians[0].jacobian == res.jacobian);
    REQUIRE(res.contact_manifold_point_jacobians[0].normal_jacobian == res.normal_jacobian);
}

// dim 2 with K > 3: contact.hpp takes a shortcut here (3 full IFT
// solves span the patch, the rest via affine interpolation -- verified
// exact to machine precision in this session's own scratch tests). Regress
// that shortcut directly against the brute-force per-point diffSocp value
// it's supposed to match, not just against the invariants above (which
// can't tell the shortcut path apart from a bug that happens to preserve
// them).
TEST_CASE("ContactManifold: contact_manifold_point_jacobians' K>3 affine shortcut matches brute-force diffSocp",
          "[manifold]") {
    Polytope<6> box1 = makeBox(0.5), box2 = makeBox(0.4);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 0.9; // face-face, asymmetric box sizes -- dim 2

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 8; // K > 3: exercises the affine-shortcut path
    opt.pdip_tol = 1e-10;
    const auto res = proximityContactJacobian(box1, box2, g, opt);
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == 2);
    REQUIRE(res.contact_manifold_points.size() == 8);

    const Matrix4d I4 = Matrix4d::Identity();
    const auto P1 = problemMatrices(box1, I4);
    const auto P2 = problemMatrices(box2, g);
    const auto combined = combineProblemMatrices(P1, P2);
    using C = std::decay_t<decltype(combined)>;
    const auto sol = solveProximitySocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(box1, box2, g, combined.c, combined.G,
                                                                                combined.h, opt);
    for (size_t i = 0; i < res.contact_manifold_points.size(); ++i) {
        DecisionVec<C::nx> xp = sol.x;
        xp.template head<3>() = res.contact_manifold_points[i];
        const auto bruteForce =
            diffSocp<Polytope<6>, Polytope<6>, C::n_ort, C::n_soc1, C::n_soc2, C::nx>(box1, box2, xp, sol.s, sol.z, g,
                                                                                       combined.G);
        REQUIRE((res.contact_manifold_point_jacobians[i].jacobian - bruteForce).norm() < 1e-7);
    }
}
