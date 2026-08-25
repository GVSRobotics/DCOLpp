// Self-comparison: dcolpp::socp's default (least-squares + bring2cone)
// solver initialization vs. a geometric one seeded from each shape's
// precomputed inner/outer bounding-sphere radii (geometric_init.hpp),
// re-targeting the cold-start scheme from the iDCOL manuscript (Sec.
// III.A/D) to DCOL's [p;alpha;extras] decision vector. Same ergodic
// pose sweep and statistics as bench_ergodic.cpp (see its header comment),
// but both variants run in the same process on the same 100k poses per
// pair, back-to-back -- this is DCOL++ against itself, not against Julia
// (already settled, DEVIATIONS.md §1c).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/geometric_init.hpp"
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

struct Stats {
    double avg_us = 0.0, median_us = 0.0, avg_iters = 0.0;
    std::size_t succ = 0, failed = 0;
};

static Stats summarize(std::vector<double>& durations_us, std::vector<int>& iters_used, int N) {
    Stats st;
    st.succ = durations_us.size();
    st.failed = static_cast<std::size_t>(N) - st.succ;
    if (st.succ > 0) {
        st.avg_us = std::accumulate(durations_us.begin(), durations_us.end(), 0.0) / static_cast<double>(st.succ);
        std::sort(durations_us.begin(), durations_us.end());
        st.median_us = durations_us[st.succ / 2];
        st.avg_iters = std::accumulate(iters_used.begin(), iters_used.end(), 0.0) / static_cast<double>(st.succ);
    }
    return st;
}

template <typename Shape1, typename Shape2>
void runCase(const char* name, const Shape1& s1, const Shape2& s2, int N, double t_max, double r_min, double r_max) {
    using Clock = std::chrono::steady_clock;
    const double dt = t_max / N;
    const Matrix4d I4 = Matrix4d::Identity();
    SocpOptions opt;
    opt.pdip_tol = 1e-10;

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
    using C = std::decay_t<decltype(problems[0])>;

    // -- generic (default) init --
    std::vector<double> dur_generic;
    std::vector<int> iters_generic;
    dur_generic.reserve(N);
    iters_generic.reserve(N);
    for (int i = 0; i < 10; ++i) {
        const auto& c = problems[i];
        solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, opt);
    }
    for (int i = 0; i < N; ++i) {
        const auto& c = problems[i];
        const auto t0 = Clock::now();
        const auto sol = solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, opt);
        const auto t1 = Clock::now();
        if (!sol.converged) continue;
        dur_generic.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters_generic.push_back(sol.iters);
    }

    // -- geometric init --
    std::vector<double> dur_geo;
    std::vector<int> iters_geo;
    dur_geo.reserve(N);
    iters_geo.reserve(N);
    for (int i = 0; i < 10; ++i) {
        const auto& c = problems[i];
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, x0);
        solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, opt, &init);
    }
    for (int i = 0; i < N; ++i) {
        const auto& c = problems[i];
        const auto t0 = Clock::now();
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, x0);
        const auto sol = solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(c.c, c.G, c.h, opt, &init);
        const auto t1 = Clock::now();
        if (!sol.converged) continue;
        dur_geo.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters_geo.push_back(sol.iters);
    }

    const Stats sg = summarize(dur_generic, iters_generic, N);
    const Stats sh = summarize(dur_geo, iters_geo, N);

    std::cout << "CASE " << name
              << " | generic avg_us=" << sg.avg_us << " median_us=" << sg.median_us
              << " avg_iters=" << sg.avg_iters << " succ=" << sg.succ << "/" << N
              << " || geometric avg_us=" << sh.avg_us << " median_us=" << sh.median_us
              << " avg_iters=" << sh.avg_iters << " succ=" << sh.succ << "/" << N
              << " || speedup(avg)=" << (sg.avg_us / sh.avg_us) << "x\n";
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
