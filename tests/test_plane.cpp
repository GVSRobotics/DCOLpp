// Plane (half-space): shape 1, Plane(normal, point). Fixed surface n.p = d
// (d = normal.point) -- the plane does not scale; only body 2 does, so
// alpha = |n.c - d| / radius. If body 2's centre is on the -normal side the
// row is negated and plane_flipped is set (the contact normal then points
// along -normal, toward body 2). Checks the alpha law, the flip + normal
// sign, the witness on the fixed plane, d(alpha)/dxi vs FD, the box-flat 2D
// manifold, and a tilted normal.

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp"
#include "dcolpp/socp/proximity.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

// body-2 pose: centre at world (x, y, z), optional rotation w.
Matrix4d poseAt(double x, double y = 0.0, double z = 0.0, const Vector3d& w = Vector3d::Zero()) {
    Vector6d xi;
    xi << w, Vector3d::Zero();
    Matrix4d g = dcolpp::se3::Exp(xi);
    g(0, 3) += x;
    g(1, 3) += y;
    g(2, 3) += z;
    return g;
}

template <typename S1, typename S2>
double fdAlpha(const S1& s1, const S2& s2, const Matrix4d& g, int k, double eps = 1e-5) {
    Vector6d ep = Vector6d::Zero(), em = Vector6d::Zero();
    ep(k) = eps;
    em(k) = -eps;
    const double ap = proximity(s1, s2, Matrix4d(g * dcolpp::se3::Exp(ep))).alpha;
    const double am = proximity(s1, s2, Matrix4d(g * dcolpp::se3::Exp(em))).alpha;
    return (ap - am) / (2 * eps);
}

} // namespace

TEST_CASE("Plane: alpha = |signed distance| / radius, both sides", "[plane]") {
    const Plane ground; // n = +x, d = 1
    const Sphere ball(0.5);

    struct Case { double cx, alpha; bool flip; };
    for (const Case& c : {Case{2.5, 3.0, false}, Case{1.5, 1.0, false}, Case{1.25, 0.5, false},
                          Case{0.75, 0.5, true}, Case{0.5, 1.0, true}, Case{-1.0, 4.0, true}}) {
        const auto r = proximityContact(ground, ball, poseAt(c.cx));
        const Vector3d exp_n = c.flip ? Vector3d(-1, 0, 0) : Vector3d(1, 0, 0);
        REQUIRE(r.converged);
        REQUIRE(std::abs(r.alpha - c.alpha) < 1e-5);
        REQUIRE(r.plane_flipped == c.flip);
        REQUIRE((r.normal - exp_n).norm() < 1e-5);
        REQUIRE(std::abs(r.witness_point.x() - 1.0) < 1e-5); // witness on the fixed plane
        if (c.cx > 1.5 + 1e-9) REQUIRE(r.alpha > 1.0);
        if (c.cx > 1.0 && c.cx < 1.5 - 1e-9) REQUIRE(r.alpha < 1.0);
    }
}

TEST_CASE("Plane: constant normal, invariant to body-2 rotation and lateral slide", "[plane]") {
    const Plane ground;
    const Cylinder cyl(0.5, 1.4);
    const Vector3d tilt(0.3, -0.5, 0.7);

    for (double y : {-2.0, 0.0, 1.5}) {
        const auto r = proximityContact(ground, cyl, poseAt(2.0, y, -0.4, tilt));
        REQUIRE(r.converged);
        REQUIRE_FALSE(r.plane_flipped);
        REQUIRE((r.normal - Vector3d::UnitX()).norm() < 1e-6);
    }
    const double a0 = proximity(ground, cyl, poseAt(2.0, 0.0, 0.0, tilt)).alpha;
    const double a1 = proximity(ground, cyl, poseAt(2.0, 3.0, -2.0, tilt)).alpha;
    REQUIRE(std::abs(a0 - a1) < 1e-6);
}

TEST_CASE("Plane: d(alpha)/dxi matches central FD", "[plane]") {
    const Plane ground;
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    const Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    const Polytope<6> cube(A, b);

    for (const Vector3d& w : {Vector3d(0, 0, 0), Vector3d(0.2, -0.1, 0.15), Vector3d(-0.3, 0.25, 0.1)}) {
        for (double x : {2.1, 1.7, 1.35}) { // stay off the sd = 0 kink
            const Matrix4d g = poseAt(x, 0.4, -0.2, w);
            const auto ag = alphaGradient(ground, cube, g);
            REQUIRE(ag.converged);
            for (int k = 0; k < 6; ++k) {
                const double fd = fdAlpha(ground, cube, g, k);
                REQUIRE(std::abs(ag.grad(0, k) - fd) < 1e-4);
            }
        }
    }
}

TEST_CASE("Plane: normal_jacobian is zero, witness on the fixed plane", "[plane]") {
    const Plane ground;
    const Ellipsoid ell(0.9, 0.6, 0.5);

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    const auto r = proximityContactJacobian(ground, ell, poseAt(2.0, 0.5, -0.3, Vector3d(0.2, 0.1, -0.15)), opt);
    REQUIRE(r.converged);
    REQUIRE(r.normal_jacobian.norm() < 1e-9); // normal is a constant
    REQUIRE(std::abs(r.witness_point.x() - 1.0) < 1e-5);
    REQUIRE((r.normal - Vector3d::UnitX()).norm() < 1e-6);
}

TEST_CASE("Plane: cube flat on the plane is a 2D contact manifold", "[plane]") {
    const Plane ground;
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    const Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    const Polytope<6> cube(A, b);

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 4;

    const auto r = proximityContactJacobian(ground, cube, poseAt(1.5), opt); // face flush at x = 1
    REQUIRE(r.converged);
    REQUIRE(std::abs(r.alpha - 1.0) < 1e-5);
    REQUIRE(r.contact_manifold_dim == 2);
    REQUIRE_FALSE(r.witness_jacobian_valid);
    REQUIRE(r.contact_manifold_points.size() == 4u);
    for (const auto& p : r.contact_manifold_points) {
        REQUIRE(std::abs(p.x() - 1.0) < 1e-5);
        REQUIRE(std::abs(std::abs(p.y()) - 0.5) < 0.05);
        REQUIRE(std::abs(std::abs(p.z()) - 0.5) < 0.05);
    }
}

TEST_CASE("Plane: tilted normal", "[plane]") {
    const Sphere ball(0.5);
    const Vector3d n = Vector3d(1, 1, 0).normalized();
    const Plane wall(n, 2.0 * n); // d = 2

    // centre at 2.5*n -> gap 0.5 along n -> alpha == 1, not flipped
    const auto r = proximityContact(wall, ball, poseAt(2.5 * n.x(), 2.5 * n.y(), 0.0));
    REQUIRE(r.converged);
    REQUIRE(std::abs(r.alpha - 1.0) < 1e-5);
    REQUIRE_FALSE(r.plane_flipped);
    REQUIRE((r.normal - n).norm() < 1e-5);

    // centre at 1.5*n (behind the surface) -> alpha == 1, flipped, normal -n
    const auto rf = proximityContact(wall, ball, poseAt(1.5 * n.x(), 1.5 * n.y(), 0.0));
    REQUIRE(rf.converged);
    REQUIRE(std::abs(rf.alpha - 1.0) < 1e-5);
    REQUIRE(rf.plane_flipped);
    REQUIRE((rf.normal + n).norm() < 1e-5);

    const Matrix4d g = poseAt(2.7 * n.x(), 2.7 * n.y(), 0.3);
    const auto ag = alphaGradient(wall, ball, g);
    for (int k = 0; k < 6; ++k) {
        REQUIRE(std::abs(ag.grad(0, k) - fdAlpha(wall, ball, g, k)) < 1e-4);
    }
}
