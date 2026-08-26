// ContactDegeneracy (proximity_contact.hpp): contact_manifold_dim /
// normal_cone_dim / witness_jacobian_valid / normal_jacobian_valid.
//
// These hand-built configurations were used to derive the theory itself
// (DEVIATIONS.md "Contact-manifold degeneracy"), so this file locks in that
// investigation as a permanent regression test rather than leaving it in
// scratch scripts. Each case is checked two independent ways that must
// agree: (1) contactDegeneracy's rank-based prediction, computed at a
// single (loose) pdip_tol, and (2) a direct pdip_tol sweep (1e-4 vs 1e-12)
// of the actual proximityContactJacobian output -- a >1e6x norm blowup is
// the same "diverges like 1/pdip_tol" signature used throughout
// DEVIATIONS.md, i.e. ground truth independent of contactDegeneracy's own
// machinery.
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_contact.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix3d;
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

template <typename Shape1, typename Shape2>
bool jacobianDiverges(const Shape1& s1, const Shape2& s2, const Matrix4d& g, bool witness_not_normal) {
    double n_loose = 0, n_tight = 0;
    for (double tol : {1e-4, 1e-12}) {
        SocpOptions opt;
        opt.pdip_tol = tol;
        const auto full = proximityContactJacobian(s1, s2, g, opt);
        if (!full.converged) return false;
        const double v = witness_not_normal ? full.jacobian.norm() : full.normal_jacobian.norm();
        if (tol == 1e-4) n_loose = v; else n_tight = v;
    }
    return (n_tight / n_loose) > 1e6;
}

template <typename Shape1, typename Shape2>
void checkCase(const Shape1& s1, const Shape2& s2, const Matrix4d& g, int expect_manifold_dim, int expect_cone_dim) {
    SocpOptions opt;
    opt.compute_degeneracy_info = true; // opt-in: off by default (measured ~10-20% of call cost)
    const auto res = proximityContactJacobian(s1, s2, g, opt);
    REQUIRE(res.converged);

    REQUIRE(res.contact_manifold_dim == expect_manifold_dim);
    REQUIRE(res.normal_cone_dim == expect_cone_dim);
    REQUIRE(res.witness_jacobian_valid == (expect_manifold_dim == 0));
    REQUIRE(res.normal_jacobian_valid == (expect_cone_dim == 0));

    // Cross-check against the independent pdip_tol-sweep ground truth.
    REQUIRE(jacobianDiverges(s1, s2, g, true) == !res.witness_jacobian_valid);
    REQUIRE(jacobianDiverges(s1, s2, g, false) == !res.normal_jacobian_valid);
}

} // namespace

TEST_CASE("ContactDegeneracy: off by default -- sentinel -1, not a computed 0", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0; // the parallel edge-edge case: a real, nonzero degeneracy
    const auto res = proximityContactJacobian(box, box, g); // default SocpOptions -- opted out
    REQUIRE(res.converged);
    REQUIRE(res.contact_manifold_dim == -1);
    REQUIRE(res.normal_cone_dim == -1);
    // Not computed -> booleans default optimistic ("not checked"), even
    // though this specific pose IS degenerate (checked with opt-in above).
    REQUIRE(res.witness_jacobian_valid == true);
    REQUIRE(res.normal_jacobian_valid == true);
}

TEST_CASE("ContactDegeneracy: box face-face parallel -> 2D manifold, unique normal", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    checkCase(box, box, g, /*manifold*/ 2, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: box corner-corner -> unique witness, non-unique normal", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0; g(2, 3) = 1.0;
    checkCase(box, box, g, /*manifold*/ 0, /*cone*/ 2);
}

TEST_CASE("ContactDegeneracy: box edge-edge parallel -> 1D manifold, non-unique normal", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0; g(1, 3) = 1.0;
    checkCase(box, box, g, /*manifold*/ 1, /*cone*/ 1);
}

TEST_CASE("ContactDegeneracy: box edge-edge SKEW (non-parallel) -> both unique", "[degeneracy]") {
    // Same touching point as the parallel edge-edge case, but box2 rotated
    // about the (1,1,0) diagonal (so BOTH of its edge-forming facet normals
    // tilt away from box1's -- rotating about x or y alone leaves one
    // facet's normal unchanged and silently collapses this back into a
    // face-type contact).
    Polytope<6> box1 = makeBox(0.5);
    Polytope<6> box2 = makeBox(0.5);
    for (double deg : {40.0, 90.0}) {
        const double th = deg * 3.14159265358979323846 / 180.0;
        const Matrix3d R2 = Eigen::AngleAxisd(th, Vector3d(1, 1, 0).normalized()).toRotationMatrix();
        const Vector3d touch_world(0.5, 0.5, 0.0);
        const Vector3d local_pt(-0.5, -0.5, 0.0);
        Matrix4d g = Matrix4d::Identity();
        g.block<3, 3>(0, 0) = R2;
        g.block<3, 1>(0, 3) = touch_world - R2 * local_pt;
        checkCase(box1, box2, g, /*manifold*/ 0, /*cone*/ 0);
    }
}

TEST_CASE("ContactDegeneracy: cylinder-cylinder axis-parallel -> 1D manifold, unique normal", "[degeneracy]") {
    Cylinder cyl(0.4, 1.5);
    Matrix4d g = Matrix4d::Identity();
    g(1, 3) = 0.8;
    checkCase(cyl, cyl, g, /*manifold*/ 1, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: sphere-sphere -> both unique", "[degeneracy]") {
    Sphere s1(0.8), s2(0.6);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.4;
    checkCase(s1, s2, g, /*manifold*/ 0, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: sphere vs box face -> both unique", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Sphere s(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.0;
    checkCase(box, s, g, /*manifold*/ 0, /*cone*/ 0);
}

// IsStrictlyConvex: two convex bodies touching, at least one strictly
// convex, can only touch at a single point (the boundary can't contain the
// line segment a bigger contact set would require) -- and the combined
// valid normal there is the intersection of both bodies' normal cones,
// which collapses to the smooth body's own single ray regardless of how
// degenerate the OTHER body's cone is (even a sharp vertex). So a sphere or
// ellipsoid touching a box CORNER or EDGE -- not just a face, the cases
// that actually stress this -- should still give both dims 0.
static_assert(IsStrictlyConvex<Sphere>::value);
static_assert(IsStrictlyConvex<Ellipsoid>::value);
static_assert(!IsStrictlyConvex<Capsule>::value);
static_assert(!IsStrictlyConvex<Cylinder>::value);
static_assert(!IsStrictlyConvex<Cone>::value);
static_assert(!IsStrictlyConvex<Polytope<6>>::value);
static_assert(!IsStrictlyConvex<Polygon<4>>::value);

TEST_CASE("ContactDegeneracy: sphere touching a box CORNER exactly -> both unique", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Sphere s(0.6);
    const Vector3d corner(0.5, 0.5, 0.5);
    const Vector3d n = Vector3d(1, 1, 1).normalized();
    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = corner + 0.6 * n;
    checkCase(box, s, g, /*manifold*/ 0, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: sphere touching a box EDGE exactly -> both unique", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Sphere s(0.6);
    const Vector3d edge_mid(0.5, 0.5, 0.0);
    const Vector3d n = Vector3d(1, 1, 0).normalized();
    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = edge_mid + 0.6 * n;
    checkCase(box, s, g, /*manifold*/ 0, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: ellipsoid near a box CORNER -> both unique", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix3d P = Vector3d(0.9, 0.6, 0.5).asDiagonal();
    Ellipsoid e(P);
    const Vector3d corner(0.5, 0.5, 0.5);
    const Vector3d n = Vector3d(1, 1, 1).normalized();
    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = corner + 0.9 * n;
    checkCase(box, e, g, /*manifold*/ 0, /*cone*/ 0);
}

TEST_CASE("ContactDegeneracy: ellipsoid near a box EDGE -> both unique", "[degeneracy]") {
    Polytope<6> box = makeBox(0.5);
    Matrix3d P = Vector3d(0.9, 0.6, 0.5).asDiagonal();
    Ellipsoid e(P);
    const Vector3d edge_mid(0.5, 0.5, 0.0);
    const Vector3d n = Vector3d(1, 1, 0).normalized();
    Matrix4d g = Matrix4d::Identity();
    g.block<3, 1>(0, 3) = edge_mid + 0.9 * n;
    checkCase(box, e, g, /*manifold*/ 0, /*cone*/ 0);
}
