// Timing comparison: DCOL++ vs. the original DifferentiableCollisions.jl,
// for proximityJacobian on the same shape pairs used by
// gen_socp_reference.jl/bench_socp.jl. See DEVIATIONS.md for the numbers
// this produced.
//
// Optional CMake target (a one-off comparison, not a regression test):
//   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DDCOLPP_BUILD_TESTS=OFF -DDCOLPP_BUILD_EXAMPLES=OFF -DDCOLPP_BUILD_BENCHMARKS=ON
//   cmake --build build-bench --target bench_socp
//
// If compiling manually, define NDEBUG/EIGEN_NO_DEBUG and link the non-header
// sources, e.g.:
//   g++ -O3 -DNDEBUG -DEIGEN_NO_DEBUG -std=c++17 -I include -I <eigen3 include dir> tools/bench_socp.cpp src/se3.cpp src/socp_analytic_derivatives.cpp -o bench_socp
//
// Performance experiment: -DEIGEN_DONT_VECTORIZE made this tiny fixed-size
// benchmark much faster on MinGW/GCC, but exposed robustness failures in the
// full randomized test suite. Do not enable it globally without revalidating.
#include <chrono>
#include <iostream>
#include <random>

#include "portable_random.hpp"

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomG(std::mt19937& rng) {
    dcolpp_test::PortableNormal nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.head<3>() *= 0.8;
    Matrix4d g = dcolpp::se3::Exp(xi);
    g.block<3, 1>(0, 3) += 2.0 * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

constexpr int kIters = 20000;
constexpr double kPi = 3.14159265358979323846;
volatile double g_sink = 0.0;

template <typename Shape1, typename Shape2>
void benchPair(const char* name, const Shape1& s1, const Shape2& s2, std::mt19937& rng) {
    const Matrix4d g = randomG(rng);
    SocpOptions opt;
    opt.pdip_tol = 1e-10;

    for (int i = 0; i < 10; ++i) proximityJacobian(s1, s2, g, opt);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        auto r = proximityJacobian(s1, s2, g, opt);
        if (!r.converged) std::cerr << "not converged\n"; // prevent over-optimization
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double jac_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    const auto P1 = problemMatrices(s1, Matrix4d::Identity());
    const auto P2 = problemMatrices(s2, g);
    const auto combined = combineProblemMatrices(P1, P2);
    const auto sol = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
        combined.c, combined.G, combined.h, opt);

    const auto ts0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        const auto sr = solveSocp<combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            combined.c, combined.G, combined.h, opt);
        g_sink += sr.x(3);
    }
    const auto ts1 = std::chrono::steady_clock::now();
    const double solve_us = std::chrono::duration<double, std::micro>(ts1 - ts0).count() / kIters;

    const auto td0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        const auto J = diffSocp<Shape1, Shape2, combined.n_ort, combined.n_soc1, combined.n_soc2, combined.nx>(
            s1, s2, sol.x, sol.s, sol.z, g, combined.G);
        g_sink += J(3, 0);
    }
    const auto td1 = std::chrono::steady_clock::now();
    const double diff_us = std::chrono::duration<double, std::micro>(td1 - td0).count() / kIters;

    std::cout << name << ": " << jac_us << " us/call"
              << " (solve " << solve_us << ", diff " << diff_us << ", iters " << sol.iters << ")\n";
}

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

int main() {
    std::mt19937 rng(20260821);
    benchPair("SphereSphere", Sphere(0.5), Sphere(0.3), rng);
    benchPair("SphereCapsule", Sphere(0.4), Capsule(0.3, 1.2), rng);
    benchPair("CapsuleCylinder", Capsule(0.3, 1.2), Cylinder(0.25, 0.9), rng);
    benchPair("CylinderCone", Cylinder(0.25, 0.9), Cone(1.2, 22.0 * kPi / 180.0), rng);
    benchPair("ConePolytope", Cone(1.2, 22.0 * kPi / 180.0), makeCube(), rng);
    benchPair("PolytopePolytope", makeCube(0.5), makeCube(0.35), rng);
    benchPair("PolytopeEllipsoid", makeCube(), makeEllipsoid(), rng);
    benchPair("EllipsoidPolygon", makeEllipsoid(), makeHex(), rng);
    benchPair("PolygonSphere", makeHex(), Sphere(0.35), rng);
    return 0;
}
