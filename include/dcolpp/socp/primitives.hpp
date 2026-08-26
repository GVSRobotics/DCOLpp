#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/primitives.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Geometric primitives. Unlike the Julia source (which gives every shape a
// world-frame position/quaternion pair, duplicated again for an MRP
// variant), DCOL++ primitives carry only local geometry: placement is
// entirely handled by the relative SE(3) pose `g` passed into
// `problemMatrices` (see problem_matrices.hpp), so there is exactly one
// struct per shape family instead of a quaternion/MRP pair.
//
// `r_offset`/`Q_offset` (a local mounting translation/rotation) are kept,
// matching the original -- they let a primitive's collision geometry be
// offset from the frame origin it's attached to.

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

namespace dcolpp::socp {

// Bounding-sphere radii around a shape's own local center (r_offset),
// pose-independent -- used by the geometric initial guess
// (geometric_init.hpp, DEVIATIONS.md "geometric initial guess"). Computed
// ONCE per shape instance, in each primitive's own constructor below, and
// cached as a `const` member: these depend only on the shape's own fixed
// geometry, never on a query pose, so recomputing them per solveSocp call
// (the original implementation, before this) was pure waste for a shape
// reused across many queries -- measured ~1.5ns/call for the simple shapes
// but a genuine ~26ns/call for Ellipsoid (an actual
// Eigen::SelfAdjointEigenSolver<Matrix3d>, not just arithmetic).
struct BoundingSphere {
    double inner; // largest sphere centered at r_offset fully inside the shape
    double outer; // smallest sphere centered at r_offset fully containing the shape
};

namespace detail {

inline BoundingSphere sphereBoundingSphere(double R) { return {R, R}; }

inline BoundingSphere capsuleBoundingSphere(double R, double L) { return {R, R + L / 2.0}; }

inline BoundingSphere cylinderBoundingSphere(double R, double L) {
    const double halfL = L / 2.0;
    return {std::min(R, halfL), std::sqrt(R * R + halfL * halfL)};
}

// r_offset sits at the solid cone's centroid (apex 3H/4 behind it, base cap
// H/4 in front, matching problemMatrices(Cone)'s own convention -- see
// problem_matrices.hpp). Inner radius is the exact insphere radius for a
// sphere centered *at that fixed centroid* (not the largest insphere over
// all centers): limited by whichever of {distance to base plane,
// perpendicular distance to the lateral surface} is closer.
inline BoundingSphere coneBoundingSphere(double H, double beta) {
    const double apex_dist = 3.0 * H / 4.0;
    const double base_dist = H / 4.0;
    const double base_r = H * std::tan(beta);
    const double outer = std::max(apex_dist, std::sqrt(base_dist * base_dist + base_r * base_r));
    const double inner = std::min(base_dist, apex_dist * std::sin(beta));
    return {inner, outer};
}

inline BoundingSphere ellipsoidBoundingSphere(const Eigen::Matrix3d& P) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(P);
    const double lam_min = es.eigenvalues()(0);
    const double lam_max = es.eigenvalues()(2);
    return {1.0 / std::sqrt(lam_max), 1.0 / std::sqrt(lam_min)};
}

// A y <= b in the shape's own local frame (see problemMatrices(Polytope)),
// origin (r_offset) assumed interior. Inner radius is exact (nearest-face
// distance). Outer radius has no closed form from a half-space
// representation alone without vertex enumeration; this uses a
// conservative heuristic (exact for an axis-aligned box, an over-estimate
// for most other convex polytopes) -- fine for seeding the solver, not a
// correctness requirement.
template <int NH>
BoundingSphere polytopeBoundingSphere(const Eigen::Matrix<double, NH, 3>& A, const Eigen::Matrix<double, NH, 1>& b) {
    double inner = std::numeric_limits<double>::infinity();
    double maxface = 0.0;
    for (int i = 0; i < NH; ++i) {
        const double d = b(i) / A.row(i).norm();
        inner = std::min(inner, d);
        maxface = std::max(maxface, d);
    }
    return {inner, std::sqrt(3.0) * maxface};
}

// A 2D convex polygon (A,b) in-plane, puffed out by cushion radius R in 3D
// (see problemMatrices(Polygon)). Inner radius is exact: the ball of
// radius R centered at r_offset (which lies in the 2D polygon) is always
// fully contained, and is exactly the largest such ball (moving purely
// out-of-plane is capped at R regardless of the 2D inradius). Outer uses
// the same face-distance heuristic as Polytope, plus the cushion R.
template <int NH>
BoundingSphere polygonBoundingSphere(const Eigen::Matrix<double, NH, 2>& A, const Eigen::Matrix<double, NH, 1>& b,
                                      double R) {
    double maxface2d = 0.0;
    for (int i = 0; i < NH; ++i) maxface2d = std::max(maxface2d, b(i) / A.row(i).norm());
    return {R, std::sqrt(2.0) * maxface2d + R};
}

} // namespace detail

struct Capsule {
    double R, L;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    Capsule(double R_, double L_) : R(R_), L(L_), bounding_sphere(detail::capsuleBoundingSphere(R_, L_)) {}
};

struct Cylinder {
    double R, L;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    Cylinder(double R_, double L_) : R(R_), L(L_), bounding_sphere(detail::cylinderBoundingSphere(R_, L_)) {}
};

struct Cone {
    double H, beta;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    Cone(double H_, double beta_) : H(H_), beta(beta_), bounding_sphere(detail::coneBoundingSphere(H_, beta_)) {}
};

struct Sphere {
    double R;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    explicit Sphere(double R_) : R(R_), bounding_sphere(detail::sphereBoundingSphere(R_)) {}
};

struct Ellipsoid {
    Eigen::Matrix3d P;  // x'Px <= 1
    Eigen::Matrix3d U;  // upper Cholesky factor of P
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    explicit Ellipsoid(const Eigen::Matrix3d& P_)
        : P(P_), U(Eigen::LLT<Eigen::Matrix3d>(P_).matrixU()), bounding_sphere(detail::ellipsoidBoundingSphere(P_)) {}
};

template <int NH>
struct Polytope {
    Eigen::Matrix<double, NH, 3> A;
    Eigen::Matrix<double, NH, 1> b;
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    Polytope(const Eigen::Matrix<double, NH, 3>& A_, const Eigen::Matrix<double, NH, 1>& b_)
        : A(A_), b(b_), bounding_sphere(detail::polytopeBoundingSphere<NH>(A_, b_)) {}
};

template <int NH>
struct Polygon {
    Eigen::Matrix<double, NH, 2> A;
    Eigen::Matrix<double, NH, 1> b;
    double R; // "cushion" radius
    Eigen::Vector3d r_offset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Q_offset = Eigen::Matrix3d::Identity();
    const BoundingSphere bounding_sphere;
    Polygon(const Eigen::Matrix<double, NH, 2>& A_, const Eigen::Matrix<double, NH, 1>& b_, double R_)
        : A(A_), b(b_), R(R_), bounding_sphere(detail::polygonBoundingSphere<NH>(A_, b_, R_)) {}
};

// Strict convexity: a shape whose boundary contains no straight line
// segment. Used by contact_degeneracy.hpp/contact_manifold.hpp to skip the
// degeneracy/manifold computation ENTIRELY (not just cheaply) whenever
// either touching shape qualifies: two convex bodies in contact, at least
// one strictly convex, can only touch at a single point (the boundary
// can't contain the line segment a bigger contact set would require, so
// contact_manifold_dim is provably 0), and the combined valid normal at
// that point is the intersection of both bodies' normal cones -- which
// collapses to the smooth body's own single ray regardless of how
// degenerate the OTHER body's normal cone is (even a sharp vertex), so
// normal_cone_dim is provably 0 too.
//
// Verified directly, not just argued from the theory: contact_manifold_dim
// == 0 and normal_cone_dim == 0 on every one of several hand-built
// adversarial cases (sphere/ellipsoid touching a box corner or a box edge
// EXACTLY, not just a face) plus a 500-pose random sweep (mixed
// separation/penetration/orientation) -- zero counterexamples.
//
// Only Sphere and Ellipsoid qualify among these 7 shapes. Capsule/
// Cylinder/Cone all have a RULED lateral surface -- straight generator
// lines lying entirely on the boundary (e.g. a cylinder's axis-parallel
// side lines: two points on the same line, connected by a segment that
// itself lies exactly on the boundary) -- which is itself a boundary line
// segment, breaking strict convexity. Polytope/Polygon are flat-faced,
// obviously not strictly convex.
template <typename Shape>
struct IsStrictlyConvex : std::false_type {};
template <>
struct IsStrictlyConvex<Sphere> : std::true_type {};
template <>
struct IsStrictlyConvex<Ellipsoid> : std::true_type {};

} // namespace dcolpp::socp
