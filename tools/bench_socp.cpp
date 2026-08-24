// Timing comparison: DCOL++ vs. the original DifferentiableCollisions.jl,
// for proximityJacobian on the same shape pairs used by
// gen_socp_reference.jl/bench_socp.jl. See DEVIATIONS.md for the numbers
// this produced.
//
// Not wired into the CMake build (a one-off comparison, not a regression
// test); compile standalone, e.g.:
//   g++ -O3 -std=c++17 -I include -I <eigen3 include dir> tools/bench_socp.cpp -o bench_socp
#include <chrono>
#include <iostream>
#include <random>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity.hpp"

using namespace dcolpp::socp;
using Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

Matrix4d randomG(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vector6d xi;
    for (int i = 0; i < 6; ++i) xi(i) = nd(rng);
    xi.head<3>() *= 0.8;
    Matrix4d g = dcolpp::se3::Exp(xi);
    g.block<3, 1>(0, 3) += 2.0 * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

constexpr int kIters = 20000;
constexpr double kPi = 3.14159265358979323846;

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
    const double us_per_call = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
    std::cout << name << ": " << us_per_call << " us/call\n";
}

Polytope<6> makeCube() {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
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

int main() {
    std::mt19937 rng(20260821);
    benchPair("SphereSphere", Sphere(0.5), Sphere(0.3), rng);
    benchPair("SphereCapsule", Sphere(0.4), Capsule(0.3, 1.2), rng);
    benchPair("CapsuleCylinder", Capsule(0.3, 1.2), Cylinder(0.25, 0.9), rng);
    benchPair("CylinderCone", Cylinder(0.25, 0.9), Cone(1.2, 22.0 * kPi / 180.0), rng);
    benchPair("ConePolytope", Cone(1.2, 22.0 * kPi / 180.0), makeCube(), rng);
    benchPair("PolytopeEllipsoid", makeCube(), makeEllipsoid(), rng);
    benchPair("EllipsoidPolygon", makeEllipsoid(), makeHex(), rng);
    benchPair("PolygonSphere", makeHex(), Sphere(0.35), rng);
    return 0;
}
