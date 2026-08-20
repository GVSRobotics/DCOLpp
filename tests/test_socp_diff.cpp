#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_gradient.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomG(std::mt19937& rng, double translation_scale = 2.5) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.tail<3>() *= 0.8;
    Matrix4d g = dcolpp::se3::Exp(xi);
    g.block<3, 1>(0, 3) += translation_scale * Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

// Central finite difference of [witness; alpha] w.r.t. g's local twist,
// using the exact SE(3) exponential (independent of se3::retract, which is
// what the analytic Jacobian is built on) as ground truth.
template <typename Shape1, typename Shape2>
Eigen::Matrix<double, 4, 6> fdJacobian(const Shape1& s1, const Shape2& s2, const Matrix4d& g0, double eps,
                                        const SocpOptions& opt) {
    Eigen::Matrix<double, 4, 6> J;
    for (int i = 0; i < 6; ++i) {
        Vector6d e = Vector6d::Zero();
        e(i) = eps;
        const Matrix4d gp = dcolpp::se3::SE3Compose(g0, dcolpp::se3::Exp(e));
        const Matrix4d gm = dcolpp::se3::SE3Compose(g0, dcolpp::se3::Exp(-e));
        const ProximityResult rp = proximity(s1, s2, gp, opt);
        const ProximityResult rm = proximity(s1, s2, gm, opt);
        REQUIRE(rp.converged);
        REQUIRE(rm.converged);
        J.template block<3, 1>(0, i) = (rp.witness_point - rm.witness_point) / (2.0 * eps);
        J(3, i) = (rp.alpha - rm.alpha) / (2.0 * eps);
    }
    return J;
}

template <typename Shape1, typename Shape2>
void checkDiff(const Shape1& s1, const Shape2& s2, int ntrials, unsigned seed) {
    std::mt19937 rng(seed);
    SocpOptions opt;
    opt.pdip_tol = 1e-10;
    const double eps = 1e-6;

    for (int t = 0; t < ntrials; ++t) {
        const Matrix4d g = randomG(rng);

        const ProximityJacobianResult jr = proximityJacobian(s1, s2, g, opt);
        REQUIRE(jr.converged);

        const ProximityGradientResult gr = proximityGradient(s1, s2, g, opt);
        REQUIRE(gr.converged);

        // Cross-check: the two independent differentiation paths must agree
        // on d(alpha)/dxi (jacobian's alpha row vs. the dedicated gradient).
        //
        // Tolerance note: the linear system A = G'(S\Z)G that both diffSocp
        // and the original Julia diff_socp solve can be extremely
        // ill-conditioned for some shape/pose combinations (observed
        // cond(A) ~ 1e10-1e12 for e.g. Sphere vs. a narrow Cone) -- this was
        // confirmed to be identical in the *original* Julia library (same
        // cond(A), same formula), not something the C++ port introduced.
        // The original test suite (proximity_test.jl,
        // polytope_derivs_test.jl) itself only requires finite-difference
        // agreement to ~1e-2 for exactly this reason, so 1e-2 here is
        // matching upstream's accepted tolerance, not loosening a bug away.
        INFO("trial " << t << " (jacobian row 3 vs proximityGradient)");
        REQUIRE((jr.jacobian.row(3) - gr.grad).norm() < 1e-2);

        // Finite-difference self-consistency (Julia parity isn't directly
        // comparable here: the parameterization changed from a 12/14-dof
        // world pose to a 6-dof relative twist).
        const Eigen::Matrix<double, 4, 6> fd = fdJacobian(s1, s2, g, eps, opt);
        INFO("trial " << t << " (analytic jacobian vs finite difference)");
        REQUIRE((jr.jacobian - fd).norm() < 1e-2);
    }
}

} // namespace

TEST_CASE("socp differentiation: sphere-sphere", "[socp][diff]") {
    checkDiff(Sphere(0.5), Sphere(0.3), 8, 1);
}

TEST_CASE("socp differentiation: capsule-sphere", "[socp][diff]") {
    checkDiff(Capsule(0.3, 1.2), Sphere(0.4), 8, 2);
}

TEST_CASE("socp differentiation: cylinder-cone", "[socp][diff]") {
    checkDiff(Cylinder(0.25, 0.9), Cone(1.2, 22.0 * 3.14159265358979323846 / 180.0), 8, 3);
}

TEST_CASE("socp differentiation: polytope-ellipsoid", "[socp][diff]") {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    Polytope<6> cube(A, b);

    Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
    P(0, 0) = 1.0 / (0.6 * 0.6);
    P(1, 1) = 1.0 / (0.4 * 0.4);
    P(2, 2) = 1.0 / (0.5 * 0.5);
    Ellipsoid ell(P);

    checkDiff(cube, ell, 8, 4);
}

TEST_CASE("socp differentiation: polygon-sphere", "[socp][diff]") {
    Eigen::Matrix<double, 6, 2> A;
    Eigen::Matrix<double, 6, 1> b;
    for (int i = 0; i < 6; ++i) {
        const double th = 2.0 * 3.14159265358979323846 * i / 6.0;
        A(i, 0) = std::cos(th);
        A(i, 1) = std::sin(th);
    }
    b.setConstant(0.4);
    Polygon<6> hex(A, b, 0.1);

    checkDiff(hex, Sphere(0.35), 8, 5);
}
