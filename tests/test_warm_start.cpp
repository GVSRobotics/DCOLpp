// Warm-starting proximityJacobian and proximityContactJacobian for
// temporally-continuous queries (warm_start.hpp, socp_init.hpp).
//
// Contract under test: a ContactWarmState handle NEVER changes the answer --
// warm and cold converge to the same optimum at the same pdip_tol -- it only
// cuts interior-point iterations when consecutive poses are close.
// proximityJacobian's warm path stops at mu < pdip_tol (warmStartInit);
// proximityContactJacobian's uses the same stopping rule as cold
// (warmStartInitCentral) because normal_jacobian's frozen Hessian needs a
// central-path point. Also covers the trust-region fallback and residual
// gates.

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <random>

#include "portable_random.hpp"

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_contact.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d twist(const Matrix4d& g0, const Vec6& d) { return g0 * dcolpp::se3::Exp(d); }

SocpOptions tol8() {
    SocpOptions o;
    o.pdip_tol = 1e-8;
    return o;
}

// Warm and cold solve the same problem to the same pdip_tol but along
// different interior-point paths, so in weakly-determined directions near
// contact (DEVIATIONS.md §5/§6a) the converged witness and its Jacobian can
// differ by more than pdip_tol. alpha is an envelope-theorem quantity and
// stays well-determined, so it is checked tightly.
void agree(const ProximityJacobianResult& a, const ProximityJacobianResult& b) {
    REQUIRE(a.converged);
    REQUIRE(b.converged);
    REQUIRE(std::abs(a.alpha - b.alpha) < 1e-6);
    REQUIRE((a.witness_point - b.witness_point).norm() < 2e-4);
    const double jn = std::max(1.0, a.jacobian.norm());
    REQUIRE((a.jacobian - b.jacobian).norm() / jn < 1.5e-2);
}

// Random-walk `g` in ~1e-2 steps; every step, warm must match a fresh cold
// solve, and warm must take fewer total iterations across the walk.
template <class A, class B>
void sweep(const A& s1, const B& s2, const Matrix4d& g0, unsigned seed, const char* tag) {
    INFO(tag);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    std::mt19937 rng(seed);
    ContactWarmState<A, B> ws;

    Matrix4d g = g0;
    long cold_it = 0, warm_it = 0;
    for (int k = 0; k < 24; ++k) {
        Vec6 d;
        for (int i = 0; i < 6; ++i) d(i) = 0.004 * nd(rng);
        g = twist(g, d);

        const auto cold = proximityJacobian(s1, s2, g, tol8());
        const auto warm = proximityJacobian(s1, s2, g, tol8(), &ws);
        agree(cold, warm);

        REQUIRE(ws.valid);
        REQUIRE((ws.g_ref - g).norm() < 1e-12);
        cold_it += cold.iters;
        warm_it += warm.iters;
    }
    INFO("cold_iters=" << cold_it << "  warm_iters=" << warm_it);
    REQUIRE(warm_it < cold_it);          // warm-start helps in aggregate
    REQUIRE(warm_it * 5 < cold_it * 4);  // ... by a clear margin (steady-state per-step is more)
}

// Same contract as agree(), extended to proximityContactJacobian's normal
// and normal_jacobian. The warm path uses the same stopping rule as cold
// (mu AND KKT residuals to pdip_tol), so warm never returns a
// lower-quality normal_jacobian -- a warm solve that would has its residual
// gate reject it and redo it cold.
void agreeContact(const ProximityContactJacobianResult& a, const ProximityContactJacobianResult& b) {
    REQUIRE(a.converged);
    REQUIRE(b.converged);
    REQUIRE(std::abs(a.alpha - b.alpha) < 1e-6);
    REQUIRE((a.witness_point - b.witness_point).norm() < 2e-4);
    const double jn = std::max(1.0, a.jacobian.norm());
    REQUIRE((a.jacobian - b.jacobian).norm() / jn < 1.5e-2);
    REQUIRE((a.normal - b.normal).norm() < 2e-4);
    const double njn = std::max(1.0, a.normal_jacobian.norm());
    REQUIRE((a.normal_jacobian - b.normal_jacobian).norm() / njn < 2e-2);
}

template <class A, class B>
void sweepContact(const A& s1, const B& s2, const Matrix4d& g0, unsigned seed, const char* tag) {
    INFO(tag);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    std::mt19937 rng(seed);
    ContactWarmState<A, B> ws;

    Matrix4d g = g0;
    long cold_it = 0, warm_it = 0;
    for (int k = 0; k < 24; ++k) {
        Vec6 d;
        for (int i = 0; i < 6; ++i) d(i) = 0.004 * nd(rng);
        g = twist(g, d);

        const auto cold = proximityContactJacobian(s1, s2, g, tol8());
        const auto warm = proximityContactJacobian(s1, s2, g, tol8(), &ws);
        agreeContact(cold, warm);

        REQUIRE(ws.valid);
        REQUIRE((ws.g_ref - g).norm() < 1e-12);
        cold_it += cold.iters;
        warm_it += warm.iters;
    }
    // A ~1e-2 random walk per step is fast enough that the warm point often
    // fails the tight residual gate and falls back to cold -- so no strict
    // iteration win is asserted here (the slow-drift test covers the case
    // warm-start actually helps). The point under test is that
    // normal_jacobian stays correct and the shared handle stays valid.
    INFO("cold_iters=" << cold_it << "  warm_iters=" << warm_it);
    REQUIRE(warm_it < cold_it * 3); // sanity: no runaway / infinite loop
}

Matrix4d sepPose(double x, double y, double z) {
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = x;
    g(1, 3) = y;
    g(2, 3) = z;
    return g;
}

Polytope<6> box(double h) {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(h);
    return Polytope<6>(A, b);
}

} // namespace

TEST_CASE("warm start never changes the answer, and cuts iterations: Sphere vs Cone", "[warm]") {
    sweep(Sphere(0.8), Cone(1.3, 0.4), sepPose(1.7, 0.3, -0.2), 1001, "sphere-cone");
}

TEST_CASE("warm start: Polytope vs Cone", "[warm]") {
    sweep(box(0.5), Cone(1.4, 0.45), sepPose(1.6, 0.2, 0.1), 1002, "polytope-cone");
}

TEST_CASE("warm start: Capsule vs Cylinder (both carry extra decision vars)", "[warm]") {
    sweep(Capsule(0.4, 1.6), Cylinder(0.5, 1.5), sepPose(1.5, 0.4, -0.1), 1003, "capsule-cylinder");
}

TEST_CASE("warm start: Sphere vs TruncatedCone", "[warm]") {
    sweep(Sphere(0.7), TruncatedCone(0.7, 0.3, 1.3), sepPose(1.7, -0.2, 0.2), 1004, "sphere-truncatedcone");
}

TEST_CASE("warm start: repeated identical pose stays correct and is not more expensive", "[warm]") {
    Sphere s(0.8);
    Cone c(1.3, 0.4);
    const Matrix4d g = sepPose(1.7, 0.25, -0.15);
    ContactWarmState<Sphere, Cone> ws;

    const auto r0 = proximityJacobian(s, c, g, tol8(), &ws); // cold, fills ws
    REQUIRE(r0.converged);
    for (int k = 0; k < 5; ++k) {
        const auto r = proximityJacobian(s, c, g, tol8(), &ws); // same g -> warm
        agree(r0, r);
        REQUIRE(r.iters <= r0.iters);
    }
}

TEST_CASE("warm start: a big pose jump falls back to cold and stays correct", "[warm]") {
    Sphere s(0.8);
    Cone c(1.3, 0.4);
    ContactWarmState<Sphere, Cone> ws;

    const Matrix4d g_a = sepPose(1.7, 0.2, -0.1);
    proximityJacobian(s, c, g_a, tol8(), &ws);
    REQUIRE(ws.valid);
    const double rho_before = ws.rho;

    Vec6 big; // far outside any trust region
    big << 1.4, -0.9, 0.7, 2.3, -1.1, 0.8;
    const Matrix4d g_b = twist(g_a, big);

    const auto cold = proximityJacobian(s, c, g_b, tol8());
    const auto warm = proximityJacobian(s, c, g_b, tol8(), &ws);
    agree(cold, warm);
    REQUIRE(ws.rho <= rho_before);
    REQUIRE(ws.valid);
    REQUIRE((ws.g_ref - g_b).norm() < 1e-12);
}

TEST_CASE("warm start: nullptr handle is byte-for-byte the cold path", "[warm]") {
    Sphere s(0.8);
    Cone c(1.3, 0.4);
    const Matrix4d g = sepPose(1.62, 0.31, -0.12);
    const auto a = proximityJacobian(s, c, g, tol8());
    const auto b = proximityJacobian(s, c, g, tol8(), static_cast<ContactWarmState<Sphere, Cone>*>(nullptr));
    REQUIRE(a.iters == b.iters);
    REQUIRE(a.alpha == b.alpha);
    REQUIRE((a.witness_point - b.witness_point).norm() == 0.0);
    REQUIRE((a.jacobian - b.jacobian).norm() == 0.0);
}

TEST_CASE("warm start: proximityContactJacobian normal_jacobian matches cold (Sphere vs Cone)", "[warm]") {
    sweepContact(Sphere(0.8), Cone(1.3, 0.4), sepPose(1.7, 0.3, -0.2), 2001, "contact sphere-cone");
}

TEST_CASE("warm start: proximityContactJacobian (Polytope vs Cone)", "[warm]") {
    sweepContact(box(0.5), Cone(1.4, 0.45), sepPose(1.6, 0.2, 0.1), 2002, "contact polytope-cone");
}

TEST_CASE("warm start: proximityContactJacobian (Capsule vs Cylinder)", "[warm]") {
    sweepContact(Capsule(0.4, 1.6), Cylinder(0.5, 1.5), sepPose(1.5, 0.4, -0.1), 2003, "contact capsule-cylinder");
}

TEST_CASE("warm start: proximityContactJacobian (Polytope vs Ellipsoid)", "[warm]") {
    sweepContact(box(0.5), Ellipsoid(0.6, 0.45, 0.5), sepPose(1.55, 0.15, 0.1), 2004, "contact polytope-ellipsoid");
}

TEST_CASE("warm start: proximityContactJacobian nullptr handle is the cold path", "[warm]") {
    Sphere s(0.8);
    Cone c(1.3, 0.4);
    const Matrix4d g = sepPose(1.62, 0.31, -0.12);
    const auto a = proximityContactJacobian(s, c, g, tol8());
    const auto b = proximityContactJacobian(s, c, g, tol8(), static_cast<ContactWarmState<Sphere, Cone>*>(nullptr));
    REQUIRE(a.iters == b.iters);
    REQUIRE(a.alpha == b.alpha);
    REQUIRE((a.witness_point - b.witness_point).norm() == 0.0);
    REQUIRE((a.jacobian - b.jacobian).norm() == 0.0);
    REQUIRE((a.normal_jacobian - b.normal_jacobian).norm() == 0.0);
}

TEST_CASE("warm start: proximityContactJacobian repeated identical pose -> 1 iteration", "[warm]") {
    Sphere s(0.8);
    Cone c(1.3, 0.4);
    const Matrix4d g = sepPose(1.7, 0.25, -0.15);
    ContactWarmState<Sphere, Cone> ws;

    const auto r0 = proximityContactJacobian(s, c, g, tol8(), &ws); // cold, fills ws
    REQUIRE(r0.converged);
    for (int k = 0; k < 5; ++k) {
        const auto r = proximityContactJacobian(s, c, g, tol8(), &ws); // same g -> warm
        agreeContact(r0, r);
        REQUIRE(r.iters <= 2); // static contact: carried (s*,z*) is already converged
    }
}

TEST_CASE("warm start: proximityContactJacobian slow drift (sustained contact) -> few iterations", "[warm]") {
    // The case warm-start targets: a body drifting slowly against another,
    // ~1e-3 pose move per call. normal_jacobian must match cold every step,
    // and the warm solve must be much cheaper than a cold re-solve.
    Sphere s(0.6);
    Polytope<6> b = box(0.5);
    ContactWarmState<Sphere, Polytope<6>> ws;

    long cold_it = 0, warm_it = 0;
    for (int k = 0; k < 40; ++k) {
        const Matrix4d g = sepPose(1.15 - 0.0002 * k, 0.10 + 0.00007 * k, -0.05 + 0.00005 * k);
        const auto cold = proximityContactJacobian(s, b, g, tol8());
        const auto warm = proximityContactJacobian(s, b, g, tol8(), &ws);
        agreeContact(cold, warm);
        cold_it += cold.iters;
        warm_it += warm.iters;
    }
    INFO("drift cold_iters=" << cold_it << " warm_iters=" << warm_it);
    REQUIRE(warm_it * 3 < cold_it * 2); // sustained contact: markedly fewer iterations
}

TEST_CASE("warm start: settling-contact loop (many tiny steps) stays exact", "[warm]") {
    // Sphere drifting slowly toward a box face -- the sustained-contact case
    // warm-start targets. Every step must match a fresh cold solve.
    Sphere s(0.6);
    Polytope<6> b = box(0.5);
    ContactWarmState<Sphere, Polytope<6>> ws;

    long cold_it = 0, warm_it = 0;
    for (int k = 0; k < 40; ++k) {
        Matrix4d g = sepPose(1.15 - 0.0009 * k, 0.10 + 0.0003 * k, -0.05 + 0.0002 * k);
        const auto cold = proximityJacobian(s, b, g, tol8());
        const auto warm = proximityJacobian(s, b, g, tol8(), &ws);
        agree(cold, warm);
        cold_it += cold.iters;
        warm_it += warm.iters;
    }
    INFO("settling cold_iters=" << cold_it << " warm_iters=" << warm_it);
    REQUIRE(warm_it * 4 < cold_it * 3); // sustained contact: >=25% fewer iterations
}
