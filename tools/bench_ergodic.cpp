// Ergodic sweep benchmark for dcolpp::socp's solve pipeline (problem-matrix
// construction + solveSocp; no differentiation), mirroring iDCOL's
// examples/ergodic.cpp pose generator and statistics -- same quasi-random,
// deterministic pose coverage (7 incommensurate frequencies over t) instead
// of one fixed pose repeated many times, so timing isn't overfit to one
// PDIP iteration count/conditioning. See tools/bench_ergodic.jl for the
// Julia-side counterpart used for comparison.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/socp_init.hpp"
#include "dcolpp/socp/proximity.hpp"

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

static double percentile_sorted(const std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::size_t idx = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size())));
    idx = std::min(idx, v.size());
    idx = (idx == 0) ? 0 : idx - 1;
    return v[idx];
}

template <typename Shape1, typename Shape2>
void runCase(const char* name, const Shape1& s1, const Shape2& s2, int N, double t_max, double r_min, double r_max) {
    using Clock = std::chrono::steady_clock;
    const double dt = t_max / N;
    const Matrix4d I4 = Matrix4d::Identity();
    SocpOptions opt;
    opt.pdip_tol = 1e-10;

    // Pre-generate all N (c,G,h) problems from the ergodic pose sweep,
    // untimed -- isolates the solve (what "solving" means here) from
    // problem-matrix construction, while still exercising it across 100k
    // diverse poses/conditionings instead of one fixed problem repeated.
    // Poses are kept too: the geometric init (default as of DEVIATIONS.md
    // §1d) needs shape1/shape2/g to build its guess, not just (c,G,h).
    const auto P1 = problemMatrices(s1, I4);
    using Combined = decltype(combineProblemMatrices(P1, problemMatrices(s2, I4)));
    std::vector<Combined> problems;
    std::vector<Matrix4d> poses;
    problems.reserve(N);
    poses.reserve(N);
    for (int i = 0; i < N; ++i) {
        const Matrix4d g = getSystematicPose(i * dt, r_min, r_max);
        const auto P2 = problemMatrices(s2, g);
        problems.push_back(combineProblemMatrices(P1, P2));
        poses.push_back(g);
    }

    std::vector<double> durations_us;
    durations_us.reserve(N);
    std::vector<int> iters_used;
    iters_used.reserve(N);
    std::size_t failed = 0;

    // Warm up (JIT-analog: touch every code path once, not timed).
    // std::decay_t<decltype(combined)>:: rather than combined:: -- GCC
    // accepts a dependent object expression directly in a template-argument
    // position here, but clang (correctly, this is the conforming
    // two-phase-lookup rule) does not.
    for (int i = 0; i < 10; ++i) {
        const auto& combined = problems[i];
        using C = std::decay_t<decltype(combined)>;
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
    }

    for (int i = 0; i < N; ++i) {
        const auto& combined = problems[i];
        using C = std::decay_t<decltype(combined)>;

        // Timed region includes the geometric guess itself (bounding-radii
        // lookup, the least-squares dual projection) -- matching exactly
        // what a real proximity()/proximityJacobian() call now does by
        // default, not just the bare solveSocp iteration loop.
        const auto t0 = Clock::now();
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        const auto sol = solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
        const auto t1 = Clock::now();

        if (!sol.converged) {
            failed++;
            continue;
        }
        durations_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters_used.push_back(sol.iters);
    }

    const std::size_t succ = durations_us.size();
    double avg_us = 0.0, median_us = 0.0, p25 = 0.0, p75 = 0.0, p95 = 0.0, stddev = 0.0, avg_iters = 0.0;
    if (succ > 0) {
        avg_us = std::accumulate(durations_us.begin(), durations_us.end(), 0.0) / succ;
        std::sort(durations_us.begin(), durations_us.end());
        median_us = durations_us[succ / 2];
        p25 = percentile_sorted(durations_us, 25.0);
        p75 = percentile_sorted(durations_us, 75.0);
        p95 = percentile_sorted(durations_us, 95.0);
        double sq = 0.0;
        for (double v : durations_us) sq += (v - avg_us) * (v - avg_us);
        stddev = std::sqrt(sq / succ);
        avg_iters = std::accumulate(iters_used.begin(), iters_used.end(), 0.0) / static_cast<double>(succ);
    }

    std::cout << "CASE " << name
              << " | avg_us=" << avg_us
              << " | median_us=" << median_us
              << " | p25_us=" << p25
              << " | p75_us=" << p75
              << " | p95_us=" << p95
              << " | stddev_us=" << stddev
              << " | avg_iters=" << avg_iters
              << " | success=" << (100.0 * succ / N) << "% (" << succ << " succ, " << failed << " failed)\n";
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

Ellipsoid makeEllipsoid() { return Ellipsoid(0.6, 0.4, 0.5); }
} // namespace

int main(int argc, char** argv) {
    const int N = 100000;
    const double t_max = 100.0;
    const double r_min = 0.05, r_max = 2.0;

    // One pair per process (see run_all_ergodic.sh) -- matches
    // bench_ergodic.jl's per-process isolation, for a symmetric protocol on
    // both sides even though C++ (no GC) didn't show cross-pair pollution.
    // With no argument, runs all 9 in-process (the pre-fix behavior).
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
