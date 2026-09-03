// PolytopeX (runtime face count) must reproduce the fixed-size Polytope<NH>
// path: values, full contact Jacobian, and the face-face contact manifold.
// See include/dcolpp/socp/runtime_poly/.

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp"
#include "dcolpp/socp/runtime_poly/runtime_poly.hpp"

using namespace Eigen;
namespace socp = dcolpp::socp;
namespace rpoly = dcolpp::socp::runtime_poly;

namespace {

// N unit normals on a Fibonacci sphere, b = irregular -> bounded convex
// N-face polytope with a generic (unique-witness) contact for a hull pair.
template <int N>
void fibPoly(Matrix<double, N, 3>& A, Matrix<double, N, 1>& b, double jitter) {
    const double ga = 3.14159265358979323846 * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < N; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / N;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double th = ga * i;
        A.row(i) = Vector3d(r * std::cos(th), r * std::sin(th), z).transpose();
        const double f = i * 0.6180339887 + 0.13;
        b(i) = 0.8 + jitter * (f - std::floor(f));
    }
}

Matrix4d se3g(const Vector3d& t, const Vector3d& aa) {
    Matrix4d g = Matrix4d::Identity();
    const double a = aa.norm();
    g.block<3, 3>(0, 0) = a < 1e-12 ? Matrix3d::Identity() : Matrix3d(AngleAxisd(a, aa / a));
    g.block<3, 1>(0, 3) = t;
    return g;
}

// g * Exp(xi), xi = [w; v] -- the pose-perturbation convention the analytic
// derivatives are taken against.
Matrix4d perturb(const Matrix4d& g, const Matrix<double, 6, 1>& xi) {
    Matrix4d E = Matrix4d::Identity();
    const Vector3d w = xi.head<3>(), v = xi.tail<3>();
    const double th = w.norm();
    Matrix3d R = Matrix3d::Identity();
    Vector3d tr = v;
    if (th > 1e-12) {
        const Matrix3d K = dcolpp::se3::skew<double>(Vector3d(w / th));
        const Matrix3d Sw = dcolpp::se3::skew<double>(w);
        R = Matrix3d::Identity() + std::sin(th) * K + (1 - std::cos(th)) * K * K;
        tr = (Matrix3d::Identity() + (1 - std::cos(th)) / (th * th) * Sw +
              (th - std::sin(th)) / (th * th * th) * Sw * Sw) *
             v;
    }
    E.block<3, 3>(0, 0) = R;
    E.block<3, 1>(0, 3) = tr;
    return g * E;
}

} // namespace

TEMPLATE_TEST_CASE_SIG("PolytopeX vs PolytopeX matches Polytope<N>", "[runtime_poly]", ((int N), N), 4, 6, 12, 20, 48) {
    Matrix<double, N, 3> A1, A2;
    Matrix<double, N, 1> b1, b2;
    fibPoly<N>(A1, b1, 0.35);
    fibPoly<N>(A2, b2, 0.55);
    A2 = (A2 * Matrix3d(AngleAxisd(0.7, Vector3d(0.3, 0.8, -0.5).normalized()))).eval();
    b2 *= 0.8;
    socp::Polytope<N> pf1(A1, b1), pf2(A2, b2);
    rpoly::PolytopeX px1(A1, b1), px2(A2, b2);

    std::mt19937 rng(777 + N);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double amax = 0, wpmax = 0, jmax = 0, nmax = 0, njmax = 0;
    int nc = 0;
    for (int k = 0; k < 250; ++k) {
        const double sc = (k % 3 == 0) ? 0.6 : (k % 3 == 1 ? 1.3 : 2.5);
        Vector3d t(U(rng), U(rng), U(rng));
        t = t.normalized() * sc * (0.6 + 0.4 * std::abs(U(rng)));
        Vector3d aa(U(rng), U(rng), U(rng));
        aa *= 0.9 * 3.14159 * std::abs(U(rng));
        const Matrix4d g = se3g(t, aa);

        const auto rf = socp::proximityContactJacobian(pf1, pf2, g);
        const auto rd = rpoly::proximityContactJacobian(px1, px2, g);
        REQUIRE(rf.converged == rd.converged);
        if (!rf.converged) continue;
        ++nc;
        amax = std::max(amax, std::abs(rf.alpha - rd.alpha));
        wpmax = std::max(wpmax, (rf.witness_point - rd.witness_point).norm());
        jmax = std::max(jmax, (rf.jacobian - rd.jacobian).cwiseAbs().maxCoeff());
        nmax = std::max(nmax, (rf.normal - rd.normal).cwiseAbs().maxCoeff());
        njmax = std::max(njmax, (rf.normal_jacobian - rd.normal_jacobian).cwiseAbs().maxCoeff());
    }
    REQUIRE(nc > 200);
    CHECK(amax < 1e-9);
    CHECK(wpmax < 1e-8);
    CHECK(jmax < 1e-6);
    CHECK(nmax < 1e-9);
    CHECK(njmax < 1e-5);
}

TEST_CASE("PolytopeX Jacobian matches finite difference on a clean contact", "[runtime_poly]") {
    Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Matrix<double, 6, 1> b;
    b.setConstant(0.5);
    rpoly::PolytopeX x1(A, b), x2(A, b);

    double emax = 0;
    for (int trial = 0; trial < 5; ++trial) {
        const Matrix4d g = se3g(Vector3d(0.80 + 0.06 * trial, 0.12, -0.07), Vector3d(0.55, 0.30, 0.18));
        const auto jd = rpoly::proximityContactJacobian(x1, x2, g);
        REQUIRE(jd.converged);
        const double h = 1e-6;
        Matrix<double, 4, 6> fd;
        for (int c = 0; c < 6; ++c) {
            Matrix<double, 6, 1> e = Matrix<double, 6, 1>::Zero();
            e(c) = h;
            const auto rp = rpoly::proximityContact(x1, x2, perturb(g, e));
            const auto rm = rpoly::proximityContact(x1, x2, perturb(g, Matrix<double, 6, 1>(-e)));
            fd.block<3, 1>(0, c) = (rp.witness_point - rm.witness_point) / (2 * h);
            fd(3, c) = (rp.alpha - rm.alpha) / (2 * h);
        }
        emax = std::max(emax, (fd - jd.jacobian).cwiseAbs().maxCoeff());
    }
    CHECK(emax < 5e-5);
}

TEMPLATE_TEST_CASE_SIG("Plane vs PolytopeX matches Plane vs Polytope<N>", "[runtime_poly]", ((int N), N), 6, 20) {
    Matrix<double, N, 3> A;
    Matrix<double, N, 1> b;
    fibPoly<N>(A, b, 0.4);
    socp::Polytope<N> pf(A, b);
    rpoly::PolytopeX px(A, b);
    socp::Plane plane(Vector3d(0.1, 0.2, 1.0), Vector3d(0, 0, 0));

    std::mt19937 rng(31 + N);
    std::uniform_real_distribution<double> U(-1, 1);
    double amax = 0, jmax = 0, njmax = 0;
    int nc = 0, flip = 0;
    for (int k = 0; k < 300; ++k) {
        Vector3d t(U(rng), U(rng), 0.3 + 1.6 * std::abs(U(rng)) * ((k % 5 == 0) ? -0.6 : 1.0));
        Vector3d aa(U(rng), U(rng), U(rng));
        aa *= 3.14159 * std::abs(U(rng));
        const Matrix4d g = se3g(t, aa);
        const auto rf = socp::proximityContactJacobian(plane, pf, g);
        const auto rd = rpoly::proximityContactJacobian(plane, px, g);
        REQUIRE(rf.converged == rd.converged);
        if (!rf.converged) continue;
        ++nc;
        flip += rd.plane_flipped;
        amax = std::max(amax, std::abs(rf.alpha - rd.alpha));
        jmax = std::max(jmax, (rf.jacobian - rd.jacobian).cwiseAbs().maxCoeff());
        njmax = std::max(njmax, (rf.normal_jacobian - rd.normal_jacobian).cwiseAbs().maxCoeff());
    }
    REQUIRE(nc > 150);
    REQUIRE(flip > 0); // exercised both plane-flip branches
    CHECK(amax < 1e-9);
    CHECK(jmax < 1e-6);
    CHECK(njmax < 1e-5);
}

TEST_CASE("PolytopeX vs curved primitive matches Polytope<N>", "[runtime_poly]") {
    constexpr int N = 12;
    Matrix<double, N, 3> A;
    Matrix<double, N, 1> b;
    fibPoly<N>(A, b, 0.4);
    socp::Polytope<N> hf(A, b);
    rpoly::PolytopeX hd(A, b);

    auto sweep = [&](auto prim, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> U(-1, 1);
        double amax = 0, jmax = 0, njmax = 0;
        int nc = 0;
        for (int k = 0; k < 200; ++k) {
            const double sc = (k % 3 == 0) ? 0.7 : (k % 3 == 1 ? 1.4 : 2.6);
            Vector3d t(U(rng), U(rng), U(rng));
            t = t.normalized() * sc * (0.6 + 0.4 * std::abs(U(rng)));
            Vector3d aa(U(rng), U(rng), U(rng));
            aa *= 0.9 * 3.14159 * std::abs(U(rng));
            const Matrix4d g = se3g(t, aa);
            const auto rf = socp::proximityContactJacobian(hf, prim, g);
            const auto rd = rpoly::proximityContactJacobian(hd, prim, g);
            REQUIRE(rf.converged == rd.converged);
            if (!rf.converged) continue;
            ++nc;
            amax = std::max(amax, std::abs(rf.alpha - rd.alpha));
            jmax = std::max(jmax, (rf.jacobian - rd.jacobian).cwiseAbs().maxCoeff());
            njmax = std::max(njmax, (rf.normal_jacobian - rd.normal_jacobian).cwiseAbs().maxCoeff());
        }
        REQUIRE(nc > 120);
        CHECK(amax < 1e-8);
        CHECK(jmax < 5e-6);
        CHECK(njmax < 5e-5);
    };
    sweep(socp::Sphere(0.6), 11);
    sweep(socp::Ellipsoid(0.7, 0.4, 0.55), 13);
    sweep(socp::Cone(0.9, 0.5), 15);
    sweep(socp::TruncatedCone(0.6, 0.3, 0.8), 16);
    sweep(socp::Capsule(0.35, 0.9), 18);   // nx = 5
    sweep(socp::Cylinder(0.4, 0.8), 19);   // nx = 5
}

TEST_CASE("PolytopeX face-face contact manifold matches Polytope<6>", "[runtime_poly]") {
    Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Matrix<double, 6, 1> b;
    b.setConstant(0.5);
    socp::Polytope<6> cf(A, b);
    rpoly::PolytopeX cd(A, b);

    socp::SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 4;

    for (const Matrix4d& g : {se3g(Vector3d(0.98, 0.07, -0.04), Vector3d::Zero()),
                              se3g(Vector3d(0.90, -0.03, 0.05), Vector3d::Zero()),
                              se3g(Vector3d(0.95, 0.0, 0.0), Vector3d(3.14159 / 4, 0, 0))}) {
        const auto rf = socp::proximityContactJacobian(cf, cf, g, opt);
        const auto rd = rpoly::proximityContactJacobian(cd, cd, g, opt);
        REQUIRE(rf.converged);
        REQUIRE(rd.converged);
        CHECK(rf.contact_manifold_dim == rd.contact_manifold_dim);
        CHECK(rf.normal_cone_dim == rd.normal_cone_dim);
        REQUIRE(rf.contact_manifold_points.size() == rd.contact_manifold_points.size());
        // manifold-point set agreement (same greedy selection -> same order)
        double sd = 0, jd = 0;
        for (size_t i = 0; i < rd.contact_manifold_points.size(); ++i) {
            double md = 1e300;
            int bi = 0;
            for (size_t k = 0; k < rf.contact_manifold_points.size(); ++k) {
                const double d = (rd.contact_manifold_points[i] - rf.contact_manifold_points[k]).norm();
                if (d < md) { md = d; bi = int(k); }
            }
            sd = std::max(sd, md);
            jd = std::max(jd, (rd.contact_manifold_point_jacobians[i] -
                               rf.contact_manifold_point_jacobians[bi].jacobian)
                                  .cwiseAbs()
                                  .maxCoeff());
        }
        CHECK(sd < 1e-9);
        CHECK(jd < 1e-5);
    }
}

TEST_CASE("PolytopeX warm-start matches cold and cuts iterations", "[runtime_poly]") {
    Matrix<double, 20, 3> A1, A2;
    Matrix<double, 20, 1> b1, b2;
    fibPoly<20>(A1, b1, 0.35);
    fibPoly<20>(A2, b2, 0.55);
    A2 = (A2 * Matrix3d(AngleAxisd(0.7, Vector3d(0.3, 0.8, -0.5).normalized()))).eval();
    rpoly::PolytopeX x1(A1, b1), x2(A2, b2);
    socp::Plane plane(Vector3d(0.05, 0.1, 1.0), Vector3d::Zero());

    rpoly::ContactWarmStateX wh, wp;
    double da = 0;
    long cold_it = 0, warm_it = 0;
    const int steps = 400;
    for (int k = 0; k < steps; ++k) {
        const double s = k / double(steps);
        const Matrix4d gh = se3g(Vector3d(1.12 * std::cos(0.8 * s), 1.12 * std::sin(0.8 * s), 0.08 * std::sin(1.5 * s)),
                                 Vector3d(0.15 * s, 0.22 * s, 0.1 * s));
        const Matrix4d gp = se3g(Vector3d(0.10 * std::sin(1.2 * s), 0.07 * std::cos(1.2 * s), 0.63 + 0.03 * std::sin(2 * s)),
                                 Vector3d(0.12 * s, -0.05 * s, 0.08 * s));
        const auto ch = rpoly::proximity(x1, x2, gh);
        const auto wh_r = rpoly::proximity(x1, x2, gh, socp::SocpOptions{}, &wh);
        const auto cp = rpoly::proximity(plane, x1, gp);
        const auto wp_r = rpoly::proximity(plane, x1, gp, socp::SocpOptions{}, &wp);
        REQUIRE(ch.converged == wh_r.converged);
        REQUIRE(cp.converged == wp_r.converged);
        if (ch.converged) { cold_it += ch.iters; warm_it += wh_r.iters; da = std::max(da, std::abs(ch.alpha - wh_r.alpha)); }
        if (cp.converged) { cold_it += cp.iters; warm_it += wp_r.iters; da = std::max(da, std::abs(cp.alpha - wp_r.alpha)); }
    }
    CHECK(da < 5e-5);                                  // within the Standard warm gate
    CHECK(warm_it < static_cast<long>(0.6 * cold_it)); // >= ~1.7x fewer solver iterations
}

TEST_CASE("PolytopeX vs primitive: degeneracy dims + line manifold match Polytope<6>", "[runtime_poly]") {
    Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Matrix<double, 6, 1> b;
    b.setConstant(0.5);
    socp::Polytope<6> hf(A, b);
    rpoly::PolytopeX hd(A, b);
    socp::Cylinder cyl(0.3, 0.8); // +x cap at local x = +0.4

    socp::SocpOptions opt;
    opt.compute_contact_manifold = true;
    opt.contact_manifold_points = 4;

    for (const Matrix4d& g : {se3g(Vector3d(0.90 + 0.4, 0.04, -0.02), Vector3d::Zero()),
                              se3g(Vector3d(0.83 + 0.4, -0.05, 0.03), Vector3d::Zero()),
                              se3g(Vector3d(0.88 + 0.4, 0.0, 0.0), Vector3d(0.0, 0.10, 0.0))}) {
        const auto rf = socp::proximityContactJacobian(hf, cyl, g, opt);
        const auto rd = rpoly::proximityContactJacobian(hd, cyl, g, opt);
        REQUIRE(rf.converged);
        REQUIRE(rd.converged);
        CHECK(rf.contact_manifold_dim == rd.contact_manifold_dim);
        CHECK(rf.normal_cone_dim == rd.normal_cone_dim);
        CHECK(rf.witness_jacobian_valid == rd.witness_jacobian_valid);
        if (rd.contact_manifold_dim == 1) { // shared line -> full point set
            REQUIRE(rf.contact_manifold_points.size() == rd.contact_manifold_points.size());
            double sd = 0;
            for (const auto& p : rd.contact_manifold_points) {
                double md = 1e300;
                for (const auto& q : rf.contact_manifold_points) md = std::min(md, (p - q).norm());
                sd = std::max(sd, md);
            }
            CHECK(sd < 1e-6);
        }
    }
}
