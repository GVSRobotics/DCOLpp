// Warm-start benchmark: cold vs. warm proximityJacobian (forward solve +
// the first-order IFT Jacobian), across shape pairs and two kinds of pose
// stream -- a random walk (i.i.d. per-step twist) and a smooth ergodic
// sweep. Reports per-call wall time and interior-point iteration count.
// This is the reproducible source of DEVIATIONS.md section 9g.
//
// Optional CMake target (needs -DDCOLPP_BUILD_BENCHMARKS=ON):
//   cmake --build build --target bench_warm_start
// Or by hand:
//   clang++ -O3 -DNDEBUG -DEIGEN_NO_DEBUG -std=c++17 -I include -I <eigen> \
//     tools/bench_warm_start.cpp src/se3.cpp src/socp_analytic_derivatives.cpp \
//     -o bench_warm_start
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

namespace {
volatile double g_sink = 0.0;
constexpr int kWarm = 1000;
constexpr int kIters = 40000;

Matrix4d base(double x, double y, double z) {
    Matrix4d g = Matrix4d::Identity();
    g(0, 3) = x;
    g(1, 3) = y;
    g(2, 3) = z;
    return g;
}

// Smooth quasi-periodic twist: six incommensurate sinusoids (rotation amp
// 0.18, translation 0.35). g(t) = g0 * Exp(xiOfT(t)).
Vec6 xiOfT(double t) {
    static const double f[6] = {1.0000, 0.6180, 0.4142, 0.3090, 0.2360, 0.1732};
    static const double ph[6] = {0.0, 1.3, 2.7, 0.9, 2.1, 0.4};
    Vec6 xi;
    for (int k = 0; k < 3; ++k) xi(k) = 0.18 * std::sin(f[k] * t + ph[k]);
    for (int k = 0; k < 3; ++k) xi(3 + k) = 0.35 * std::sin(f[3 + k] * t + ph[3 + k]);
    return xi;
}

struct Row {
    double cold_ns, warm_ns, cold_it, warm_it;
    long fallback_pct;
    int warm_min, warm_max;
};

template <class A, class B>
Row measure(const A& s1, const B& s2, const std::vector<Matrix4d>& gs) {
    SocpOptions opt;
    opt.pdip_tol = 1e-8;

    for (int k = 0; k < kWarm; ++k) {
        auto r = proximityJacobian(s1, s2, gs[k], opt);
        g_sink += r.alpha;
    }
    long cit = 0;
    auto c0 = std::chrono::steady_clock::now();
    for (int k = 0; k < kIters; ++k) {
        auto r = proximityJacobian(s1, s2, gs[k], opt);
        g_sink += r.alpha;
        cit += r.iters;
    }
    auto c1 = std::chrono::steady_clock::now();

    ContactWarmState<A, B> ws;
    for (int k = 0; k < kWarm; ++k) {
        auto r = proximityJacobian(s1, s2, gs[k], opt, &ws);
        g_sink += r.alpha;
    }
    long wit = 0, fb = 0;
    int mn = 999, mx = 0;
    auto w0 = std::chrono::steady_clock::now();
    for (int k = 0; k < kIters; ++k) {
        const double rho_before = ws.rho;
        auto r = proximityJacobian(s1, s2, gs[k], opt, &ws);
        g_sink += r.alpha;
        wit += r.iters;
        mn = std::min(mn, r.iters);
        mx = std::max(mx, r.iters);
        if (ws.rho < rho_before - 1e-15) ++fb; // rho only shrinks on a cold fallback
    }
    auto w1 = std::chrono::steady_clock::now();

    return {std::chrono::duration<double, std::nano>(c1 - c0).count() / kIters,
            std::chrono::duration<double, std::nano>(w1 - w0).count() / kIters,
            static_cast<double>(cit) / kIters,
            static_cast<double>(wit) / kIters,
            fb * 100 / kIters,
            mn,
            mx};
}

void print(const char* tag, const Row& r) {
    std::printf("  %-24s | cold %6.0f ns %5.2f it | warm %6.0f ns %5.2f it  (%d..%d, fb %ld%%) | %.2fx\n", tag,
               r.cold_ns, r.cold_it, r.warm_ns, r.warm_it, r.warm_min, r.warm_max, r.fallback_pct,
               r.cold_ns / r.warm_ns);
}

template <class A, class B>
void allStreams(const A& s1, const B& s2, const Matrix4d& g0, const char* tag) {
    std::printf("\n== %s ==\n", tag);

    // constant g (body at rest / fixed-base grasp)
    {
        std::vector<Matrix4d> gs(kIters + kWarm, g0);
        print("const g", measure(s1, s2, gs));
    }
    // random walk, i.i.d. per-step twist
    for (double mag : {1e-4, 1e-3, 1e-2}) {
        std::mt19937 rng(1234);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<Matrix4d> gs;
        gs.reserve(kIters + kWarm);
        Matrix4d g = g0;
        for (int k = 0; k < kIters + kWarm; ++k) {
            gs.push_back(g);
            Vec6 d;
            for (int i = 0; i < 6; ++i) d(i) = mag * nd(rng);
            g = g * dcolpp::se3::Exp(d);
        }
        char lbl[48];
        std::snprintf(lbl, sizeof lbl, "random walk  dxi~%.0e", mag);
        print(lbl, measure(s1, s2, gs));
    }
    // smooth ergodic sweep, three traversal speeds
    for (double dt : {2.5e-5, 2.5e-4, 2.5e-3}) {
        std::vector<Matrix4d> gs;
        gs.reserve(kIters + kWarm);
        for (int k = 0; k < kIters + kWarm; ++k) gs.push_back(g0 * dcolpp::se3::Exp(xiOfT(k * dt)));
        char lbl[48];
        std::snprintf(lbl, sizeof lbl, "ergodic sweep  dt=%.1e", dt);
        print(lbl, measure(s1, s2, gs));
    }
}
} // namespace

int main() {
    allStreams(Sphere(0.8), Sphere(0.6), base(1.55, 0.0, 0.0), "Sphere vs Sphere");
    allStreams(Sphere(0.8), Cone(1.3, 0.4), base(1.65, 0.1, -0.1), "Sphere vs Cone");
    {
        Eigen::Matrix<double, 6, 3> A;
        A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(0.5);
        allStreams(Polytope<6>(A, b), Cone(1.4, 0.45), base(1.55, 0.1, 0.1), "Polytope<6> vs Cone");
    }
    allStreams(Capsule(0.4, 1.6), Cylinder(0.5, 1.5), base(1.5, 0.2, -0.1), "Capsule vs Cylinder");
    allStreams(Sphere(0.7), TruncatedCone(0.7, 0.3, 1.3), base(1.6, -0.1, 0.1), "Sphere vs TruncatedCone");
    allStreams(Cone(1.4, 0.4), Ellipsoid(0.9, 0.6, 0.5), base(1.7, 0.1, -0.1), "Cone vs Ellipsoid");
    std::printf("\n(sink=%g)\n", static_cast<double>(g_sink));
    return 0;
}
