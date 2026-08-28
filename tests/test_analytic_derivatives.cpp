// Cross-checks the analytic derivative path (Stage 1-3 -> combineXiJacobian
// -> diffSocpSensitivityAnalyticAuto -> proximityGradientAnalytic) against
// central finite differences of the exact, non-autodiff functions it claims
// to differentiate -- no autodiff anywhere in these ground truths.
//
// Two levels per case:
//  - formula level: q = d(h-Gx)/dxi, dR1/dxi = d(G^Tz)/dxi (x,z frozen at
//    the converged point), FD'd directly off problemMatrices --
//    tight tolerance, since this differentiates a smooth closed-form
//    function, not the solver.
//  - solve level: dx/ds/dz, FD'd off solveSocp itself at perturbed poses --
//    looser tolerance (see note below): this also exercises A = G'(S\Z)G's
//    conditioning, which can be severe near touching configurations
//    (DEVIATIONS.md).

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

Eigen::Matrix3d nonTrivialRotation() {
    // Fixed, non-identity, not axis-aligned -- exercises Q_offset != Identity.
    std::mt19937 rng(999);
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.head<3>() *= 0.8;
    return dcolpp::se3::Exp(xi).block<3, 3>(0, 0);
}

Polygon<4> unitSquarePolygon(double R) {
    Eigen::Matrix<double, 4, 2> A;
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    Eigen::Matrix<double, 4, 1> b = Eigen::Matrix<double, 4, 1>::Constant(1.0);
    return Polygon<4>(A, b, R);
}

template <typename Shape1, typename Shape2>
void checkAnalyticVsFD(const Shape1& s1, const Shape2& s2, int ntrials, unsigned seed) {
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
        constexpr int n_ort = n_ort1 + n_ort2;
        constexpr int ns = n_ort + n_soc1 + n_soc2;
        static_assert(nx == v1 + (v2 - 4), "combined nx must match the general column-layout formula");

        const auto sol = solveSocp<n_ort, n_soc1, n_soc2, nx>(combined.c, combined.G, combined.h, opt);
        REQUIRE(sol.converged);

        // Formula level: q, dR1/dxi, x and z frozen at sol.x/sol.z.
        Eigen::Matrix<double, ns, 6> q_fd;
        Eigen::Matrix<double, nx, 6> dR1_fd;
        for (int i = 0; i < 6; ++i) {
            Vector6d e = Vector6d::Zero();
            e(i) = eps;
            const auto Cp = combineProblemMatrices(P1, problemMatrices(s2, g * dcolpp::se3::Exp(e)));
            const auto Cm = combineProblemMatrices(P1, problemMatrices(s2, g * dcolpp::se3::Exp(-e)));
            q_fd.col(i) = ((Cp.h - Cp.G * sol.x) - (Cm.h - Cm.G * sol.x)) / (2.0 * eps);
            dR1_fd.col(i) = (Cp.G.transpose() * sol.z - Cm.G.transpose() * sol.z) / (2.0 * eps);
        }

        const Vec<v2> x2_local = extractShape2LocalX<v1, v2>(sol.x);
        const Vec<n_ort2> z_ort2 = sol.z.template segment<n_ort2>(n_ort1);
        const Vec<n_soc2> z_soc2 = sol.z.template segment<n_soc2>(n_ort + n_soc1);
        const auto shape2_deriv = shapeXiDerivative(s2, g, x2_local, z_ort2, z_soc2);
        const auto xi_jac = combineXiJacobian<n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(shape2_deriv, sol.z);

        INFO("trial " << t);
        REQUIRE((xi_jac.q - q_fd).norm() < 1e-6);
        REQUIRE((xi_jac.dR1_dxi - dR1_fd).norm() < 1e-6);

        // Envelope-theorem gradient identity grad = -q^T z.
        const Eigen::Matrix<double, 1, 6> grad =
            proximityGradientAnalytic<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(s1, s2, sol.x, sol.z, g);
        REQUIRE((grad.transpose() - (-(xi_jac.q.transpose() * sol.z))).norm() < 1e-12);

        // Near-touching-configuration guard before the solve-level FD check
        // below: min(lambda2) := min over every SOC block of s0-||s1|| (and
        // z0-||z1||) -- lambda2 -> 0 means a SOC block sits on its own
        // boundary, where the KKT solution map has a genuine kink and FD is
        // not a valid ground truth (DEVIATIONS.md, cond(A)/active-set
        // finding). The formula-level q/dR1 checks above are unaffected
        // (they don't re-solve) and already ran.
        double min_lambda2 = 1e300;
        if constexpr (n_soc1 > 0) {
            min_lambda2 = std::min(min_lambda2, sol.s(n_ort) - sol.s.template segment<n_soc1 - 1>(n_ort + 1).norm());
            min_lambda2 = std::min(min_lambda2, sol.z(n_ort) - sol.z.template segment<n_soc1 - 1>(n_ort + 1).norm());
        }
        if constexpr (n_soc2 > 0) {
            constexpr int off = n_ort + n_soc1;
            min_lambda2 = std::min(min_lambda2, sol.s(off) - sol.s.template segment<n_soc2 - 1>(off + 1).norm());
            min_lambda2 = std::min(min_lambda2, sol.z(off) - sol.z.template segment<n_soc2 - 1>(off + 1).norm());
        }
        if (min_lambda2 < 1e-4) continue;

        // Solve level: dx/ds/dz, re-solving at perturbed poses.
        Eigen::Matrix<double, nx, 6> dx_fd;
        Eigen::Matrix<double, ns, 6> ds_fd, dz_fd;
        for (int i = 0; i < 6; ++i) {
            Vector6d e = Vector6d::Zero();
            e(i) = eps;
            const auto Cp = combineProblemMatrices(P1, problemMatrices(s2, g * dcolpp::se3::Exp(e)));
            const auto Cm = combineProblemMatrices(P1, problemMatrices(s2, g * dcolpp::se3::Exp(-e)));
            const auto solp = solveSocp<n_ort, n_soc1, n_soc2, nx>(Cp.c, Cp.G, Cp.h, opt);
            const auto solm = solveSocp<n_ort, n_soc1, n_soc2, nx>(Cm.c, Cm.G, Cm.h, opt);
            REQUIRE(solp.converged);
            REQUIRE(solm.converged);
            dx_fd.col(i) = (solp.x - solm.x) / (2.0 * eps);
            ds_fd.col(i) = (solp.s - solm.s) / (2.0 * eps);
            dz_fd.col(i) = (solp.z - solm.z) / (2.0 * eps);
        }

        const auto sens =
            diffSocpSensitivityAnalyticAuto<Shape1, Shape2, n_ort1, n_soc1, n_ort2, n_soc2, v1, v2>(
                s1, s2, sol.x, sol.s, sol.z, g);

        // Tolerance note: A = G'(S\Z)G can be severely ill-conditioned near
        // touching configurations (cond(A) up to ~1e13 observed), and FD of
        // a re-solved point compounds that with genuine discretization
        // error near-degenerate slack rows amplify further -- a structural
        // property of the KKT system (DEVIATIONS.md), not a formula bug.
        // 0.1 matches test_socp_diff.cpp's own ds/dz FD tolerance for the
        // same reason.
        REQUIRE((sens.dx - dx_fd).norm() < 0.1);
        REQUIRE((sens.ds - ds_fd).norm() < 0.1);
        REQUIRE((sens.dz - dz_fd).norm() < 0.1);

        // Cross-check the production API, which wraps the same call.
        const auto jacobian_via_diffSocp = diffSocp<Shape1, Shape2, n_ort, n_soc1, n_soc2, nx>(s1, s2, sol.x, sol.s, sol.z, g);
        REQUIRE((jacobian_via_diffSocp - sens.dx.template topRows<4>()).norm() < 1e-12);
    }
}

} // namespace

TEST_CASE("analytic derivatives vs FD: Sphere vs Sphere", "[analytic]") {
    checkAnalyticVsFD(Sphere(1.3), Sphere(0.7), 20, 200);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid vs Sphere", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P; // SPD
    checkAnalyticVsFD(Ellipsoid(P), Sphere(0.6), 20, 201);
}

TEST_CASE("analytic derivatives vs FD: Cone vs Sphere", "[analytic]") {
    checkAnalyticVsFD(Cone(2.0, 0.4), Sphere(0.5), 20, 202);
}

TEST_CASE("analytic derivatives vs FD: Polytope vs Sphere", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Polytope<6>(A, b), Sphere(0.4), 20, 203);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Capsule", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.8), Capsule(0.5, 2.0), 20, 210);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid vs Capsule", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    checkAnalyticVsFD(Ellipsoid(P), Capsule(0.4, 1.5), 20, 211);
}

TEST_CASE("analytic derivatives vs FD: Polytope vs Capsule", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Polytope<6>(A, b), Capsule(0.3, 1.2), 20, 212);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Cylinder", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.8), Cylinder(0.5, 2.0), 20, 220);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid vs Cylinder", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    checkAnalyticVsFD(Ellipsoid(P), Cylinder(0.4, 1.5), 20, 221);
}

TEST_CASE("analytic derivatives vs FD: Polytope vs Cylinder", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Polytope<6>(A, b), Cylinder(0.3, 1.2), 20, 222);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Cone", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.8), Cone(2.0, 0.4), 20, 230);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid vs Cone", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    checkAnalyticVsFD(Ellipsoid(P), Cone(1.5, 0.3), 20, 231);
}

TEST_CASE("analytic derivatives vs FD: Polytope vs Cone", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Polytope<6>(A, b), Cone(1.8, 0.5), 20, 232);
}

TEST_CASE("analytic derivatives vs FD: Capsule with non-identity Q_offset", "[analytic]") {
    Capsule cap(0.5, 2.0);
    cap.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), cap, 20, 240);
}

TEST_CASE("analytic derivatives vs FD: Cylinder with non-identity Q_offset", "[analytic]") {
    Cylinder cyl(0.5, 2.0);
    cyl.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), cyl, 20, 241);
}

TEST_CASE("analytic derivatives vs FD: Cone with non-identity Q_offset", "[analytic]") {
    Cone cone(2.0, 0.4);
    cone.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), cone, 20, 242);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs TruncatedCone", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.8), TruncatedCone(0.9, 0.35, 1.6), 20, 243);
}

TEST_CASE("analytic derivatives vs FD: Polytope vs TruncatedCone", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Polytope<6>(A, b), TruncatedCone(0.8, 0.3, 1.4), 20, 244);
}

TEST_CASE("analytic derivatives vs FD: TruncatedCone vs Sphere (shape 1)", "[analytic]") {
    checkAnalyticVsFD(TruncatedCone(1.0, 0.4, 1.5), Sphere(0.6), 20, 245);
}

TEST_CASE("analytic derivatives vs FD: TruncatedCone with non-identity Q_offset", "[analytic]") {
    TruncatedCone frustum(0.9, 0.3, 1.5);
    frustum.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), frustum, 20, 246);
}

TEST_CASE("analytic derivatives vs FD: TruncatedCone (R_top -> 0, cone limit)", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.7), TruncatedCone(0.9, 1e-3, 1.6), 20, 247);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Ellipsoid", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    checkAnalyticVsFD(Sphere(0.8), Ellipsoid(P), 20, 250);
}

TEST_CASE("analytic derivatives vs FD: Cone vs Ellipsoid", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.2, 0.2, 0.1, 0.2, 1.1, 0.0, 0.1, 0.0, 0.8;
    P = P.transpose() * P;
    checkAnalyticVsFD(Cone(1.8, 0.4), Ellipsoid(P), 20, 251);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid with non-identity Q_offset", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    Ellipsoid ell(P);
    ell.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), ell, 20, 252);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Polytope", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Sphere(0.8), Polytope<6>(A, b), 20, 260);
}

TEST_CASE("analytic derivatives vs FD: Cone vs Polytope", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Cone(1.6, 0.4), Polytope<6>(A, b), 20, 261);
}

TEST_CASE("analytic derivatives vs FD: Polytope with non-identity Q_offset", "[analytic]") {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    Polytope<6> poly(A, b);
    poly.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), poly, 20, 262);
}

TEST_CASE("analytic derivatives vs FD: Sphere vs Polygon", "[analytic]") {
    checkAnalyticVsFD(Sphere(0.8), unitSquarePolygon(0.2), 20, 270);
}

TEST_CASE("analytic derivatives vs FD: Cone vs Polygon", "[analytic]") {
    checkAnalyticVsFD(Cone(1.6, 0.4), unitSquarePolygon(0.25), 20, 271);
}

TEST_CASE("analytic derivatives vs FD: Polygon with non-identity Q_offset", "[analytic]") {
    Polygon<4> poly = unitSquarePolygon(0.2);
    poly.Q_offset = nonTrivialRotation();
    checkAnalyticVsFD(Sphere(0.7), poly, 20, 272);
}

TEST_CASE("analytic derivatives vs FD: Capsule vs Sphere (shape 1 has extras)", "[analytic]") {
    checkAnalyticVsFD(Capsule(0.5, 2.0), Sphere(0.7), 20, 280);
}

TEST_CASE("analytic derivatives vs FD: Cylinder vs Cone (shape 1 has extras)", "[analytic]") {
    checkAnalyticVsFD(Cylinder(0.6, 1.8), Cone(1.5, 0.35), 20, 281);
}

TEST_CASE("analytic derivatives vs FD: Polygon vs Ellipsoid (shape 1 has extras)", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    checkAnalyticVsFD(unitSquarePolygon(0.2), Ellipsoid(P), 20, 282);
}

TEST_CASE("analytic derivatives vs FD: Capsule vs Capsule (both shapes have extras)", "[analytic]") {
    checkAnalyticVsFD(Capsule(0.5, 2.0), Capsule(0.4, 1.5), 20, 283);
}

TEST_CASE("analytic derivatives vs FD: Cylinder vs Polygon (both shapes have extras)", "[analytic]") {
    checkAnalyticVsFD(Cylinder(0.6, 1.8), unitSquarePolygon(0.25), 20, 284);
}

TEST_CASE("analytic derivatives vs FD: Polygon vs Capsule (both shapes have extras, both >1 wide)", "[analytic]") {
    checkAnalyticVsFD(unitSquarePolygon(0.2), Capsule(0.4, 1.5), 20, 285);
}

TEST_CASE("analytic derivatives vs FD: Ellipsoid vs Polytope", "[analytic]") {
    Eigen::Matrix3d P;
    P << 1.5, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 2.0;
    P = P.transpose() * P;
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(1.0);
    checkAnalyticVsFD(Ellipsoid(P), Polytope<6>(A, b), 20, 293);
}
