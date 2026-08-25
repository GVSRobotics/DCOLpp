// Surrogate-scaling + geometric-init combined experiment.
//
// iDCOL manuscript Sec. III.B (eq. 12-13): rescale the relative translation
// so the bounding spheres sit at a fixed, well-conditioned separation
// (alpha_min,S = fS) before solving, then map the solution back by a single
// scalar. Verified directly for DCOL++'s own SOCP (not just assumed from
// iDCOL's different formulation) before building this: for a rescaled
// translation d_S = d/k (k = alpha_min/fS), the *entire* decision vector
// scales exactly, x*(d_S) = x*(d)/k... equivalently x*(d) = k*x*(d_S), and
// -- a DCOL-specific finding not stated in the iDCOL paper -- the dual z*
// is exactly SCALE-INVARIANT (unaffected by k), because DCOL's objective
// c=e4 is fixed, not rescaled, unlike iDCOL's own Lagrangian. Confirmed
// numerically (Sphere-Capsule, k in {0.5,1,2,3}: x*/k bit-identical, z*
// identical to ~1e-6 PDIP-tolerance noise).
//
// Motivation for trying this on top of the existing geometric+mu0-targeted
// init (geometric_init.hpp): mu0=1e-10 there is an ABSOLUTE target, but the
// SOCP's natural scale is ~alpha -- so the same mu0 means a different
// relative precision for a near-touching pose (alpha small) than a widely
// separated one (alpha large), and the fallback path's fixed constants
// (margin_frac=0.05, kOrtFloor=0.05) have the same issue. Solving at a
// normalized surrogate scale (alpha_S ~ fS ~ O(1) always) could make both
// consistent across the whole ergodic sweep instead of only helping
// poses/pairs that happen to land near unit scale already.
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

// Rescales g's translation so alpha_min(g_S) == fS, i.e. the outer
// bounding spheres sit at separation fS*(r1_out+r2_out). Returns {g_S, k}
// where the true solution is recovered as alpha* = k*alpha_S*, x* = k*x_S*
// (whole decision vector), z* = z_S* (unchanged). Assumes r_offset==0 for
// both shapes (true for every shape used in this benchmark) so shape1's
// center sits exactly at the origin and shape2's at g's raw translation.
template <typename Shape1, typename Shape2>
std::pair<Matrix4d, double> surrogatePose(const Shape1& shape1, const Shape2& shape2, const Matrix4d& g, double fS) {
    const BoundingSphere b1 = boundingSphere(shape1);
    const BoundingSphere b2 = boundingSphere(shape2);
    const Eigen::Vector3d d = g.block<3, 1>(0, 3);
    const double dist = d.norm();
    const double alpha_min = dist / std::max(b1.outer + b2.outer, 1e-9);
    const double k = std::max(alpha_min / fS, 1e-9);
    Matrix4d g_S = g;
    g_S.block<3, 1>(0, 3) = d / k;
    return {g_S, k};
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
    constexpr double kFS = 1.0;

    std::vector<Matrix4d> poses;
    poses.reserve(N);
    for (int i = 0; i < N; ++i) poses.push_back(getSystematicPose(i * dt, r_min, r_max));

    // -- geometric init, direct (no surrogate) -- the current best from
    // geometric_init.hpp, unmodified.
    std::vector<double> dur_direct;
    std::vector<int> iters_direct;
    dur_direct.reserve(N);
    iters_direct.reserve(N);
    for (int i = 0; i < 10; ++i) {
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, poses[i]);
        const auto combined = combineProblemMatrices(P1, P2);
        using C = std::decay_t<decltype(combined)>;
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
    }
    for (int i = 0; i < N; ++i) {
        const auto P1 = problemMatrices(s1, I4);
        const auto P2 = problemMatrices(s2, poses[i]);
        const auto combined = combineProblemMatrices(P1, P2);
        using C = std::decay_t<decltype(combined)>;
        const auto t0 = Clock::now();
        const auto x0 = geometricPrimalGuess(s1, s2, poses[i]);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        const auto sol = solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
        const auto t1 = Clock::now();
        if (!sol.converged) continue;
        dur_direct.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters_direct.push_back(sol.iters);
    }

    // -- surrogate-scaled geometric init: same machinery, but solved at
    // g_S (normalized to alpha_min,S=fS), mapped back by k.
    std::vector<double> dur_surr;
    std::vector<int> iters_surr;
    dur_surr.reserve(N);
    iters_surr.reserve(N);
    for (int i = 0; i < 10; ++i) {
        const auto P1 = problemMatrices(s1, I4);
        const auto [g_S, k] = surrogatePose(s1, s2, poses[i], kFS);
        (void)k;
        const auto P2 = problemMatrices(s2, g_S);
        const auto combined = combineProblemMatrices(P1, P2);
        using C = std::decay_t<decltype(combined)>;
        const auto x0 = geometricPrimalGuess(s1, s2, g_S);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
    }
    for (int i = 0; i < N; ++i) {
        const auto P1 = problemMatrices(s1, I4);
        const auto t0 = Clock::now();
        const auto [g_S, k] = surrogatePose(s1, s2, poses[i], kFS);
        const auto P2 = problemMatrices(s2, g_S);
        const auto combined = combineProblemMatrices(P1, P2);
        using C = std::decay_t<decltype(combined)>;
        const auto x0 = geometricPrimalGuess(s1, s2, g_S);
        const auto init = initializeSocpFromGuess<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, x0);
        const auto sol = solveSocp<C::n_ort, C::n_soc1, C::n_soc2, C::nx>(combined.c, combined.G, combined.h, opt, &init);
        const auto t1 = Clock::now();
        if (!sol.converged) continue;
        dur_surr.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters_surr.push_back(sol.iters);
    }

    const Stats sd = summarize(dur_direct, iters_direct, N);
    const Stats ss = summarize(dur_surr, iters_surr, N);

    std::cout << "CASE " << name
              << " | direct avg_us=" << sd.avg_us << " avg_iters=" << sd.avg_iters << " succ=" << sd.succ << "/" << N
              << " || surrogate avg_us=" << ss.avg_us << " avg_iters=" << ss.avg_iters << " succ=" << ss.succ << "/" << N
              << " || speedup(direct/surrogate)=" << (sd.avg_us / ss.avg_us) << "x\n";
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
