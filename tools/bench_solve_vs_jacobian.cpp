// Solve-only vs. solve+derivatives (proximity() vs. proximityJacobian()),
// generic vs. geometric init (SocpOptions::init_strategy, DEVIATIONS.md
// §1d) -- all four combinations, per shape pair, timing the real public API
// end to end (problemMatrices/combineProblemMatrices included -- this is
// what a caller of proximity()/proximityJacobian() actually pays, not an
// isolated solve loop). Same ergodic pose sweep as bench_ergodic.cpp (see
// its header comment) for the same reason: one lucky/unlucky fixed pose
// doesn't represent a shape pair's real PDIP iteration-count distribution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"
#include "dcolpp/socp/proximity_gradient.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;

static Matrix4d getSystematicPose(double t, double r_min, double r_max) {
    const double f1 = std::sqrt(2.0), f2 = std::sqrt(3.0), f3 = std::sqrt(5.0);
    const double f4 = std::sqrt(7.0), f5 = std::sqrt(11.0), f6 = std::sqrt(13.0), f7 = std::sqrt(17.0);
    const double TWO_PI = 2.0 * M_PI;

    const double u = 0.5 * (1.0 + std::sin(TWO_PI * f1 * t));
    const double r_val = std::cbrt(r_min * r_min * r_min + (r_max * r_max * r_max - r_min * r_min * r_min) * u);
    const double theta = (M_PI / 2.0) * std::sin(TWO_PI * f2 * t);
    const double phi = TWO_PI * f3 * t;

    Eigen::Vector3d pos(r_val * std::cos(theta) * std::cos(phi), r_val * std::cos(theta) * std::sin(phi),
                         r_val * std::sin(theta));

    const double v1 = std::sin(TWO_PI * f4 * t), v2 = std::cos(TWO_PI * f5 * t);
    const double v3 = std::sin(TWO_PI * f6 * t), v4 = std::cos(TWO_PI * f7 * t);
    Eigen::Quaterniond q(v1, v2, v3, v4);
    q.normalize();

    Matrix4d g = Matrix4d::Identity();
    g.block<3, 3>(0, 0) = q.toRotationMatrix();
    g.block<3, 1>(0, 3) = pos;
    return g;
}

static double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename Shape1, typename Shape2>
void runCase(const char* name, const Shape1& s1, const Shape2& s2, int N, double t_max, double r_min, double r_max) {
    using Clock = std::chrono::steady_clock;
    const double dt = t_max / N;

    std::vector<Matrix4d> poses;
    poses.reserve(N);
    for (int i = 0; i < N; ++i) poses.push_back(getSystematicPose(i * dt, r_min, r_max));

    for (SocpInitStrategy strat : {SocpInitStrategy::Generic, SocpInitStrategy::Geometric}) {
        SocpOptions opt;
        opt.pdip_tol = 1e-10;
        opt.init_strategy = strat;

        // solve only (proximity())
        for (int i = 0; i < 10; ++i) proximity(s1, s2, poses[i], opt);
        std::vector<double> solve_us;
        solve_us.reserve(N);
        for (int i = 0; i < N; ++i) {
            const auto t0 = Clock::now();
            const auto r = proximity(s1, s2, poses[i], opt);
            const auto t1 = Clock::now();
            if (r.converged) solve_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        // solve + derivatives (proximityJacobian())
        for (int i = 0; i < 10; ++i) proximityJacobian(s1, s2, poses[i], opt);
        std::vector<double> jac_us;
        jac_us.reserve(N);
        for (int i = 0; i < N; ++i) {
            const auto t0 = Clock::now();
            const auto r = proximityJacobian(s1, s2, poses[i], opt);
            const auto t1 = Clock::now();
            if (r.converged) jac_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        const char* label = (strat == SocpInitStrategy::Generic) ? "generic " : "geometric";
        std::cout << "CASE " << name << " [" << label << "] solve=" << mean(solve_us)
                  << "us (n=" << solve_us.size() << "/" << N << ") solve+jac=" << mean(jac_us)
                  << "us (n=" << jac_us.size() << "/" << N << ")\n";
    }
}

namespace {
constexpr double kPi = 3.14159265358979323846;

Polytope<6> makeCube(double half_side = 0.5) {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(half_side);
    return Polytope<6>(A, b);
}

Polygon<6> makeHex() {
    Eigen::Matrix<double, 6, 2> A;
    Eigen::Matrix<double, 6, 1> b;
    for (int i = 0; i < 6; ++i) {
        const double th = 2.0 * kPi * i / 6.0;
        A(i, 0) = std::cos(th);
        A(i, 1) = std::sin(th);
    }
    b.setConstant(0.4);
    return Polygon<6>(A, b, 0.1);
}

Ellipsoid makeEllipsoid() {
    Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
    P(0, 0) = 1.0 / (0.6 * 0.6);
    P(1, 1) = 1.0 / (0.4 * 0.4);
    P(2, 2) = 1.0 / (0.5 * 0.5);
    return Ellipsoid(P);
}
} // namespace

int main(int argc, char** argv) {
    const int N = 100000;
    const double t_max = 100.0;
    const double r_min = 0.05, r_max = 2.0;
    const std::string which = argc > 1 ? argv[1] : "";

    auto run = [&](const char* name, auto s1, auto s2) {
        if (which.empty() || which == name) runCase(name, s1, s2, N, t_max, r_min, r_max);
    };
    run("SphereSphere", Sphere(0.5), Sphere(0.3));
    run("SphereCapsule", Sphere(0.4), Capsule(0.3, 1.2));
    run("CapsuleCylinder", Capsule(0.3, 1.2), Cylinder(0.25, 0.9));
    run("CylinderCone", Cylinder(0.25, 0.9), Cone(1.2, 22.0 * kPi / 180.0));
    run("ConePolytope", Cone(1.2, 22.0 * kPi / 180.0), makeCube());
    run("PolytopePolytope", makeCube(0.5), makeCube(0.35));
    run("PolytopeEllipsoid", makeCube(), makeEllipsoid());
    run("EllipsoidPolygon", makeEllipsoid(), makeHex());
    run("PolygonSphere", makeHex(), Sphere(0.35));
    return 0;
}
