// proximityContact / proximityContactJacobian (contact.hpp): the
// alpha+witness_point+normal bundle, with and without Jacobians wrt xi.
//
// The underlying math (computeProximityGradient, diffSocp,
// computeContactNormalJacobian) is already verified against FD elsewhere
// (test_analytic_derivatives.cpp, test_hessian_derivatives.cpp). What these
// tests actually exercise is the wiring in contact.hpp itself --
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
#include "dcolpp/socp/contact.hpp"

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

        // Independent ground truth: a separate alphaGradient() solve +
        // contactNormal(), exactly as a caller would have composed these by
        // hand before this helper existed.
        const auto gr = alphaGradient(s1, s2, g);
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
        // does), rather than going through contact.hpp's wrappers.
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

TEST_CASE("proximityContact matches alphaGradient+contactNormal: Sphere vs Sphere", "[contact]") {
    checkProximityContact(Sphere(1.3), Sphere(0.7), 15, 700);
}

TEST_CASE("proximityContact matches alphaGradient+contactNormal: Capsule vs Cone", "[contact]") {
    checkProximityContact(Capsule(0.5, 2.0), Cone(1.5, 0.35), 15, 701);
}

TEST_CASE("proximityContact matches alphaGradient+contactNormal: Polytope vs Ellipsoid", "[contact]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(0.6);
    checkProximityContact(Polytope<6>(A, b), Ellipsoid(0.9, 1.1, 1.4), 15, 702);
}

TEST_CASE("proximityContact matches alphaGradient+contactNormal: Polygon vs Cylinder", "[contact]") {
    checkProximityContact(unitSquarePolygon(0.2), Cylinder(0.6, 1.8), 15, 703);
}

TEST_CASE("proximityContact matches alphaGradient+contactNormal: Sphere vs TruncatedCone", "[contact]") {
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

// --- per-body scale-back witnesses + signed gap ------------------------

TEST_CASE("scale-back witnesses: sphere-sphere closed form + gap sign", "[contact]") {
    const Sphere a(0.7), b(0.5);
    for (double dx : {2.4, 1.2, 0.9}) { // separated, touching, penetrating
        Matrix4d g = Matrix4d::Identity();
        g(0, 3) = dx;
        g(1, 3) = 0.5;
        const auto r = proximityContact(a, b, g);
        REQUIRE(r.converged);
        const Vector3d rv = g.block<3, 1>(0, 3);
        const Vector3d rh = rv.normalized();
        REQUIRE((r.witness_body1 - 0.7 * rh).norm() < 1e-6);          // on sphere a
        REQUIRE((r.witness_body2 - (rv - 0.5 * rh)).norm() < 1e-6);   // on sphere b
        REQUIRE(std::abs(r.gap - (rv.norm() - 1.2)) < 1e-5);          // ||r|| - R1 - R2
        if (dx > 1.2 + 1e-6) REQUIRE(r.gap > 0);
        if (dx < 1.2 - 1e-6) REQUIRE(r.gap < 0);
    }
    // touching: both witnesses collapse onto x*
    Matrix4d gt = Matrix4d::Identity();
    gt(0, 3) = 1.2;
    const auto rt = proximityContact(a, b, gt);
    REQUIRE(std::abs(rt.alpha - 1.0) < 1e-6);
    REQUIRE((rt.witness_body1 - rt.witness_point).norm() < 1e-5);
    REQUIRE((rt.witness_body2 - rt.witness_point).norm() < 1e-5);
    REQUIRE(std::abs(rt.gap) < 1e-5);
}

TEST_CASE("scale-back witness Jacobians are the exact chain rule of res.jacobian", "[contact]") {
    // The scale-back is p1 = x*/a, p2 = r + (x*-r)/a, gap = (1-1/a)||r||, so
    // its Jacobians are a closed-form linear map of res.jacobian (= d[x*;a]/dxi)
    // and dr/dxi. Verify contact.hpp assembles that map correctly -- FD of the
    // whole pipeline would only re-test the (known ~1e-3-accurate) d(x*)/dxi.
    std::mt19937 rng(9001);
    auto check = [&](auto s1, auto s2, int nposes) {
        for (int i = 0; i < nposes; ++i) {
            const Matrix4d g = randomG(rng);
            const auto r = proximityContactJacobian(s1, s2, g);
            REQUIRE(r.converged);

            const double a = r.alpha, inv = 1.0 / a, inv2 = inv * inv;
            const Vector3d x = r.witness_point, rr = g.block<3, 1>(0, 3);
            const Eigen::Matrix<double, 3, 6> Jx = r.jacobian.topRows(3);
            const Eigen::Matrix<double, 1, 6> Ja = r.jacobian.row(3).eval();
            const Eigen::Matrix<double, 3, 6> Dr = dcolpp::se3::dPointDXi(g, Vector3d::Zero());
            const double rn = rr.norm();

            const Eigen::Matrix<double, 3, 6> dp1 = inv * Jx - x * (inv2 * Ja);
            const Eigen::Matrix<double, 3, 6> dp2 = Dr + inv * (Jx - Dr) - (x - rr) * (inv2 * Ja);
            const Eigen::Matrix<double, 1, 6> dgap = (rn * inv2) * Ja + (1.0 - inv) * ((rr.transpose() / rn) * Dr);

            REQUIRE((r.witness_body1_jacobian - dp1).norm() < 1e-10);
            REQUIRE((r.witness_body2_jacobian - dp2).norm() < 1e-10);
            REQUIRE((r.gap_jacobian - dgap).norm() < 1e-10);
        }
    };
    check(Sphere(0.7), Sphere(0.5), 6);
    check(Sphere(0.8), Capsule(0.4, 1.6), 6);
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    const Eigen::Matrix<double, 6, 1> b6 = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    check(Polytope<6>(A, b6), Cone(1.3, 0.4), 6);
}

TEST_CASE("scale-back witnesses: end-to-end FD on a well-conditioned pair", "[contact]") {
    // Sphere-sphere d(x*)/dxi is well conditioned, so the full pipeline can be
    // FD-checked here as a sanity cross-check of the chain-rule test above.
    const Sphere a(0.7), b(0.5);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 2.1;
    g(1, 3) = 0.6;
    g(2, 3) = -0.4;
    const auto r = proximityContactJacobian(a, b, g);
    const double eps = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vector6d ep = Vector6d::Zero(), em = Vector6d::Zero();
        ep(k) = eps;
        em(k) = -eps;
        const auto rp = proximityContact(a, b, Matrix4d(g * dcolpp::se3::Exp(ep)));
        const auto rm = proximityContact(a, b, Matrix4d(g * dcolpp::se3::Exp(em)));
        REQUIRE(((rp.witness_body1 - rm.witness_body1) / (2 * eps) - r.witness_body1_jacobian.col(k)).norm() < 1e-4);
        REQUIRE(((rp.witness_body2 - rm.witness_body2) / (2 * eps) - r.witness_body2_jacobian.col(k)).norm() < 1e-4);
        REQUIRE(std::abs((rp.gap - rm.gap) / (2 * eps) - r.gap_jacobian(0, k)) < 1e-4);
    }
}

TEST_CASE("scale-back witnesses: per manifold point (cube on cube)", "[contact]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    const Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    const Polytope<6> cube(A, b);
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = 1.06; // parallel faces, gap 0.06

    SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 4;
    const auto r = proximityContactJacobian(cube, cube, g, opt);
    REQUIRE(r.converged);
    REQUIRE(r.contact_manifold_dim == 2);
    REQUIRE(r.contact_manifold_witnesses.size() == r.contact_manifold_points.size());
    for (const auto& w : r.contact_manifold_witnesses) {
        REQUIRE(std::abs(w.body1.x() - 0.5) < 1e-4);   // cube-1 +x face
        REQUIRE(std::abs(w.body2.x() - 0.56) < 1e-4);  // cube-2 -x face, at x = 1.06 - 0.5
        REQUIRE(std::abs(w.gap - 0.06) < 1e-4);
    }
}
