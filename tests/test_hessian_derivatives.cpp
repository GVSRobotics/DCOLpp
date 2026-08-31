// Phase E: d^2(alpha)/dxi^2 and d(contact normal)/dxi.
//
// Verification strategy, matching test_analytic_derivatives.cpp's own:
// each shape's closed-form *HessianFrozen (H_frozen, x/z held fixed) is
// checked against central-FD of computeProximityGradient itself (which
// takes x,z,g as independent arguments -- no re-solve needed, since
// H_frozen's whole point is "how grad changes with g alone"); the full
// Hessian (computeProximityHessian, which adds the two IFT
// sensitivity cross-terms) and the contact-normal Jacobian are checked
// against central-FD of the *re-solved* gradient/normal -- the only
// ground truth that actually exercises how x*,z* move with xi.

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>
#include "portable_random.hpp"

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp"
#include "dcolpp/socp/proximity.hpp"
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

Polygon<4> unitSquarePolygon(double R) {
    Eigen::Matrix<double, 4, 2> A;
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    Eigen::Matrix<double, 4, 1> b = Eigen::Matrix<double, 4, 1>::Constant(1.0);
    return Polygon<4>(A, b, R);
}

} // namespace

// =============================================================================
// Part 2: per-shape H_frozen (hessianFrozenFull) vs. central-FD of
// computeProximityGradient itself, x and z held fixed -- no SOCP solve
// needed, since H_frozen is defined exactly as "how grad changes with g
// alone, x/z frozen at whatever values are passed in".
// =============================================================================

namespace {

template <typename Shape1, typename Shape2>
void checkHessianFrozen(const Shape1& s1, const Shape2& s2, int ntrials, unsigned seed) {
    std::mt19937 rng(seed);
    SocpOptions opt;
    opt.pdip_tol = 1e-10;
    const double eps = 1e-6;

    for (int t = 0; t < ntrials; ++t) {
        const Matrix4d g = randomG(rng);
        const Matrix4d I4 = Matrix4d::Identity();

        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;
        constexpr int n_soc1 = decltype(combined)::n_soc1;
        constexpr int n_soc2 = decltype(combined)::n_soc2;
        constexpr int nx = decltype(combined)::nx;

        const auto sol = solveSocp<n_ort1 + n_ort2, n_soc1, n_soc2, nx>(combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);

        const Eigen::Matrix<double, 6, 6> H_analytic =
            hessianFrozenFull<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x, sol.z, g);

        Eigen::Matrix<double, 6, 6> H_fd;
        for (int j = 0; j < 6; ++j) {
            Vector6d d = Vector6d::Zero();
            d(j) = 1.0;
            const Matrix4d gp = g * dcolpp::se3::Exp(Vector6d(eps * d));
            const Matrix4d gm = g * dcolpp::se3::Exp(Vector6d(-eps * d));
            const Eigen::Matrix<double, 1, 6> gradp =
                computeProximityGradient<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x,
                                                                                                    sol.z, gp);
            const Eigen::Matrix<double, 1, 6> gradm =
                computeProximityGradient<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x,
                                                                                                    sol.z, gm);
            H_fd.col(j) = ((gradp - gradm) / (2.0 * eps)).transpose();
        }

        INFO("trial " << t);
        REQUIRE((H_analytic - H_fd).norm() < 1e-4);
    }
}

} // namespace

TEST_CASE("H_frozen matches FD: Sphere vs Sphere", "[hessian]") { checkHessianFrozen(Sphere(1.3), Sphere(0.7), 15, 500); }

TEST_CASE("H_frozen matches FD: Sphere vs Capsule", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), Capsule(0.5, 2.0), 15, 501);
}

TEST_CASE("H_frozen matches FD: Sphere vs Cylinder", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), Cylinder(0.6, 1.8), 15, 502);
}

TEST_CASE("H_frozen matches FD: Sphere vs Cone", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), Cone(1.5, 0.35), 15, 503);
}

TEST_CASE("H_frozen matches FD: Sphere vs Ellipsoid", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), Ellipsoid(0.82, 0.63, 1.15), 15, 504);
}

TEST_CASE("H_frozen matches FD: Sphere vs Polytope", "[hessian]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkHessianFrozen(Sphere(0.8), Polytope<6>(A, b), 15, 505);
}

TEST_CASE("H_frozen matches FD: Sphere vs Polygon", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), unitSquarePolygon(0.2), 15, 506);
}

TEST_CASE("H_frozen matches FD: Capsule vs Cone (shape 1 has extras)", "[hessian]") {
    checkHessianFrozen(Capsule(0.5, 2.0), Cone(1.5, 0.35), 15, 520);
}

TEST_CASE("H_frozen matches FD: Sphere vs TruncatedCone", "[hessian]") {
    checkHessianFrozen(Sphere(0.8), TruncatedCone(0.9, 0.35, 1.6), 15, 522);
}

TEST_CASE("H_frozen matches FD: Polygon vs Capsule (BOTH shapes have extras)", "[hessian]") {
    checkHessianFrozen(unitSquarePolygon(0.2), Capsule(0.4, 1.5), 15, 521);
}

// =============================================================================
// Part 3: the full Hessian (computeProximityHessian) vs. central-FD of the
// *re-solved* gradient -- the only ground truth that exercises dx*/dxi,
// dz*/dxi. Tolerance: ~1e-5 relative, matching the already-established Phase
// E finding (H_frozen alone is off by ~44% here; the two IFT cross-terms are
// real, not noise).
// =============================================================================

namespace {

template <typename Shape1, typename Shape2>
void checkFullHessian(const Shape1& s1, const Shape2& s2, int ntrials, unsigned seed) {
    std::mt19937 rng(seed);
    SocpOptions opt;
    opt.pdip_tol = 1e-12;
    const double eps = 1e-5;

    auto resolveAndGrad = [&](const Matrix4d& g) {
        const Matrix4d I4 = Matrix4d::Identity();
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;
        constexpr int n_soc1 = decltype(combined)::n_soc1;
        constexpr int n_soc2 = decltype(combined)::n_soc2;
        constexpr int nx = decltype(combined)::nx;
        const auto sol = solveSocp<n_ort1 + n_ort2, n_soc1, n_soc2, nx>(combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);
        const Eigen::Matrix<double, 1, 6> grad =
            computeProximityGradient<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x, sol.z,
                                                                                                g);
        return grad;
    };

    for (int t = 0; t < ntrials; ++t) {
        const Matrix4d g = randomG(rng);
        const Matrix4d I4 = Matrix4d::Identity();
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;
        constexpr int n_soc1 = decltype(combined)::n_soc1;
        constexpr int n_soc2 = decltype(combined)::n_soc2;
        constexpr int nx = decltype(combined)::nx;
        const auto sol = solveSocp<n_ort1 + n_ort2, n_soc1, n_soc2, nx>(combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);

        // Near-touching-configuration guard: min(lambda2) := min over every SOC
        // block of s0-||s1|| (and z0-||z1||). lambda2 -> 0 means that block sits
        // on its own boundary, where the KKT solution map has a genuine kink and
        // FD is not a valid ground truth -- confirmed eps-independent from 1e-7
        // to 1e-3 (DEVIATIONS.md SS6a); cond(A) alone does not catch this case.
        double min_lambda2 = 1e300;
        {
            const int off = n_ort1 + n_ort2;
            if constexpr (n_soc1 > 0) {
                min_lambda2 = std::min(min_lambda2, sol.s(off) - sol.s.template segment<n_soc1 - 1>(off + 1).norm());
                min_lambda2 = std::min(min_lambda2, sol.z(off) - sol.z.template segment<n_soc1 - 1>(off + 1).norm());
            }
            if constexpr (n_soc2 > 0) {
                const int off2 = off + n_soc1;
                min_lambda2 =
                    std::min(min_lambda2, sol.s(off2) - sol.s.template segment<n_soc2 - 1>(off2 + 1).norm());
                min_lambda2 =
                    std::min(min_lambda2, sol.z(off2) - sol.z.template segment<n_soc2 - 1>(off2 + 1).norm());
            }
        }
        if (min_lambda2 < 1e-4) {
            INFO("trial " << t << " skipped: min lambda2 = " << min_lambda2 << " (on a cone boundary)");
            continue;
        }

        const Eigen::Matrix<double, 6, 6> H_analytic =
            computeProximityHessian<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x, sol.s,
                                                                                               sol.z, g);

        Eigen::Matrix<double, 6, 6> H_fd;
        for (int j = 0; j < 6; ++j) {
            Vector6d d = Vector6d::Zero();
            d(j) = 1.0;
            const Matrix4d gp = g * dcolpp::se3::Exp(Vector6d(eps * d));
            const Matrix4d gm = g * dcolpp::se3::Exp(Vector6d(-eps * d));
            const Eigen::Matrix<double, 1, 6> gradp = resolveAndGrad(gp);
            const Eigen::Matrix<double, 1, 6> gradm = resolveAndGrad(gm);
            H_fd.col(j) = ((gradp - gradm) / (2.0 * eps)).transpose();
        }

        INFO("trial " << t << ", min lambda2 = " << min_lambda2);
        const double relerr = (H_analytic - H_fd).norm() / std::max(1.0, H_fd.norm());
        INFO("H_analytic:\n" << H_analytic);
        INFO("H_fd:\n" << H_fd);
        REQUIRE(relerr < 5e-3);
    }
}

} // namespace

TEST_CASE("computeProximityHessian matches FD of re-solved gradient: Sphere vs Sphere", "[hessian]") {
    checkFullHessian(Sphere(1.3), Sphere(0.7), 12, 600);
}

TEST_CASE("computeProximityHessian matches FD of re-solved gradient: Sphere vs TruncatedCone", "[hessian]") {
    checkFullHessian(Sphere(0.8), TruncatedCone(0.9, 0.35, 1.6), 12, 602);
}

TEST_CASE("computeProximityHessian matches FD of re-solved gradient: Capsule vs Cone", "[hessian]") {
    checkFullHessian(Capsule(0.5, 2.0), Cone(1.5, 0.35), 12, 601);
}

TEST_CASE("computeProximityHessian matches FD of re-solved gradient: Polytope vs Ellipsoid", "[hessian]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkFullHessian(Polytope<6>(A, b), Ellipsoid(0.82, 0.63, 1.15), 12, 602);
}

// =============================================================================
// Part 4: computeContactNormalJacobian vs. central-FD of the re-solved
// contact normal.
// =============================================================================

TEST_CASE("computeContactNormalJacobian matches FD of re-solved contact normal: Sphere vs Sphere", "[hessian]") {
    std::mt19937 rng(700);
    SocpOptions opt;
    opt.pdip_tol = 1e-12;
    const double eps = 1e-5;
    const Sphere s1(1.3);
    const Sphere s2(0.7);

    auto resolveAndNormal = [&](const Matrix4d& g) {
        const Matrix4d I4 = Matrix4d::Identity();
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;
        const Eigen::Matrix<double, 1, 6> grad =
            computeProximityGradient<Sphere, Sphere, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                s1, s2, sol.x, sol.z, g);
        return (g.block<3, 3>(0, 0) * grad.tail<3>().transpose()).normalized();
    };

    for (int t = 0; t < 12; ++t) {
        const Matrix4d g = randomG(rng);
        const Matrix4d I4 = Matrix4d::Identity();
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, g);
        const auto combined = combineProblemMatrices(P1, P2);
        const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);
        constexpr int n_ort1 = decltype(P1.G_ort)::RowsAtCompileTime;
        constexpr int v1 = decltype(P1.G_ort)::ColsAtCompileTime;
        constexpr int n_ort2 = decltype(P2.G_ort)::RowsAtCompileTime;
        constexpr int v2 = decltype(P2.G_ort)::ColsAtCompileTime;

        // Near-touching-configuration guard -- see checkFullHessian's own comment.
        const double min_lambda2 =
            std::min({sol.s(0) - sol.s.template segment<3>(1).norm(), sol.z(0) - sol.z.template segment<3>(1).norm(),
                      sol.s(4) - sol.s.template segment<3>(5).norm(), sol.z(4) - sol.z.template segment<3>(5).norm()});
        if (min_lambda2 < 1e-4) {
            INFO("trial " << t << " skipped: min lambda2 = " << min_lambda2 << " (on a cone boundary)");
            continue;
        }

        const Eigen::Matrix<double, 3, 6> dn_analytic =
            computeContactNormalJacobian<Sphere, Sphere, n_ort1, combined.n_soc1, n_ort2, combined.n_soc2, v1, v2>(
                s1, s2, sol.x, sol.s, sol.z, g);

        Eigen::Matrix<double, 3, 6> dn_fd;
        for (int j = 0; j < 6; ++j) {
            Vector6d d = Vector6d::Zero();
            d(j) = 1.0;
            const Matrix4d gp = g * dcolpp::se3::Exp(Vector6d(eps * d));
            const Matrix4d gm = g * dcolpp::se3::Exp(Vector6d(-eps * d));
            dn_fd.col(j) = (resolveAndNormal(gp) - resolveAndNormal(gm)) / (2.0 * eps);
        }

        INFO("trial " << t);
        REQUIRE((dn_analytic - dn_fd).norm() < 5e-3);
    }
}
