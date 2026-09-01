#pragma once
// dcolpp::socp -- collision primitives.
//
// Each shape is described in its own frame -- origin at 0, axes along
// x/y/z. A proximity query positions it with a pose g (rotation +
// translation); there is no separate mounting offset on the struct. A
// caller that needs one folds it into g before the call (and, for the
// Jacobian, multiplies the result by the matching constant adjoint).

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

namespace dcolpp::socp {

// Bounding-sphere radii about the shape's local origin, pose-independent --
// used by the geometric initial guess. Computed ONCE per shape instance, in
// each primitive's own constructor below, and cached as a `const` member.
struct BoundingSphere {
    double inner; // largest sphere centered at the local origin fully inside the shape
    double outer; // smallest sphere centered at the local origin fully containing the shape
};

namespace detail {

inline BoundingSphere sphereBoundingSphere(double R) { return {R, R}; }

inline BoundingSphere capsuleBoundingSphere(double R, double L) { return {R, R + L / 2.0}; }

inline BoundingSphere cylinderBoundingSphere(double R, double L) {
    const double halfL = L / 2.0;
    return {std::min(R, halfL), std::sqrt(R * R + halfL * halfL)};
}

// The local origin sits at the solid cone's centroid. Inner radius is the exact insphere radius for a
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

// TruncatedCone: local origin at the axial midpoint. The
// bottom rim (radius R_bottom, the larger one) is the farthest point.
// Inner radius is the exact insphere at that fixed midpoint: limited by
// whichever of {distance to a cap = L/2, perpendicular distance to the
// lateral surface = r_mid * cos(beta), r_mid the lateral radius at x=0} is
// closer.
inline BoundingSphere truncatedConeBoundingSphere(double R_bottom, double R_top, double L) {
    const double half_l = L / 2.0;
    const double tanb = (R_bottom - R_top) / L;
    const double cosb = 1.0 / std::sqrt(1.0 + tanb * tanb);
    const double r_mid = 0.5 * (R_bottom + R_top);
    const double inner = std::min(half_l, r_mid * cosb);
    const double outer = std::sqrt(half_l * half_l + R_bottom * R_bottom);
    return {inner, outer};
}

// Axis-aligned ellipsoid with semi-axes (a, b, c): the largest contained
// sphere has the shortest semi-axis as radius, the smallest containing
// sphere the longest.
inline BoundingSphere ellipsoidBoundingSphere(double a, double b, double c) {
    return {std::min({a, b, c}), std::max({a, b, c})};
}

// A y <= b in the shape's local frame; local origin assumed interior.
// inner: exact insphere radius (nearest-face distance).
// outer: exact circumradius. For convex polytopes, finds vertices from
// C(NH,3) plane triples where 3 faces meet and satisfy all inequalities.
// Falls back to sqrt(3)*farthest-face if no vertex found (degenerate case).
template <int NH>
BoundingSphere polytopeBoundingSphere(const Eigen::Matrix<double, NH, 3>& A, const Eigen::Matrix<double, NH, 1>& b) {
    double inner = std::numeric_limits<double>::infinity();
    double maxface = 0.0;
    for (int i = 0; i < NH; ++i) {
        const double d = b(i) / A.row(i).norm();
        inner = std::min(inner, d);
        maxface = std::max(maxface, d);
    }

    const double feas_tol = 1e-9 * (1.0 + b.cwiseAbs().maxCoeff());
    double outer = 0.0;
    for (int i = 0; i < NH; ++i) {
        for (int j = i + 1; j < NH; ++j) {
            for (int k = j + 1; k < NH; ++k) {
                Eigen::Matrix3d M;
                M.row(0) = A.row(i);
                M.row(1) = A.row(j);
                M.row(2) = A.row(k);
                const double scale = A.row(i).norm() * A.row(j).norm() * A.row(k).norm();
                if (scale <= 0.0 || std::abs(M.determinant()) < 1e-9 * scale) continue; // planes not independent
                const Eigen::Vector3d y = M.fullPivLu().solve(Eigen::Vector3d(b(i), b(j), b(k)));
                if (((A * y).array() <= b.array() + feas_tol).all()) outer = std::max(outer, y.norm());
            }
        }
    }
    if (!(outer > 0.0)) outer = std::sqrt(3.0) * maxface;

    return {inner, outer};
}

// A 2D convex polygon (A,b) in-plane, puffed out by cushion radius R in 3D
// (see problemMatrices(Polygon)).
// inner: exact -- the ball of radius R about the local origin.
// outer: exact -- (farthest 2D polygon vertex) + R. 2D vertices are the
// feasible C(NH,2) edge-line intersections (2x2 solves, done ONCE in the
// ctor). Falls back to sqrt(2)*farthest-edge + R only if no vertex is
// found (degenerate).
template <int NH>
BoundingSphere polygonBoundingSphere(const Eigen::Matrix<double, NH, 2>& A, const Eigen::Matrix<double, NH, 1>& b,
                                      double R) {
    double maxface2d = 0.0;
    for (int i = 0; i < NH; ++i) maxface2d = std::max(maxface2d, b(i) / A.row(i).norm());

    const double feas_tol = 1e-9 * (1.0 + b.cwiseAbs().maxCoeff());
    double vmax = 0.0;
    for (int i = 0; i < NH; ++i) {
        for (int j = i + 1; j < NH; ++j) {
            Eigen::Matrix2d M;
            M.row(0) = A.row(i);
            M.row(1) = A.row(j);
            const double scale = A.row(i).norm() * A.row(j).norm();
            if (scale <= 0.0 || std::abs(M.determinant()) < 1e-9 * scale) continue; // parallel edges
            const Eigen::Vector2d u = M.fullPivLu().solve(Eigen::Vector2d(b(i), b(j)));
            if (((A * u).array() <= b.array() + feas_tol).all()) vmax = std::max(vmax, u.norm());
        }
    }
    if (!(vmax > 0.0)) vmax = std::sqrt(2.0) * maxface2d;

    return {R, vmax + R};
}

} // namespace detail

struct Capsule {
    double R, L;
    const BoundingSphere bounding_sphere;
    Capsule(double R_, double L_) : R(R_), L(L_), bounding_sphere(detail::capsuleBoundingSphere(R_, L_)) {}
};

struct Cylinder {
    double R, L;
    const BoundingSphere bounding_sphere;
    Cylinder(double R_, double L_) : R(R_), L(L_), bounding_sphere(detail::cylinderBoundingSphere(R_, L_)) {}
};

struct Cone {
    double H, beta;
    const BoundingSphere bounding_sphere;
    Cone(double H_, double beta_) : H(H_), beta(beta_), bounding_sphere(detail::coneBoundingSphere(H_, beta_)) {}
};

// A right circular cone frustum: radius R_bottom at local x = -L/2, radius
// R_top at local x = +L/2, axis +x, local origin at the axial midpoint
// Requires R_bottom > R_top >= 0, L > 0.
// The lateral surface is the same infinite cone as Cone's, half-angle beta
// with tan(beta) = (R_bottom - R_top)/L; 
struct TruncatedCone {
    double R_bottom, R_top, L;
    double tan_beta;
    double apex_dist;
    const BoundingSphere bounding_sphere;
    TruncatedCone(double R_bottom_, double R_top_, double L_)
        : R_bottom(R_bottom_),
          R_top(R_top_),
          L(L_),
          tan_beta((R_bottom_ - R_top_) / L_),
          apex_dist(L_ / 2.0 + R_top_ * L_ / (R_bottom_ - R_top_)),
          bounding_sphere(detail::truncatedConeBoundingSphere(R_bottom_, R_top_, L_)) {}
};

struct Sphere {
    double R;
    const BoundingSphere bounding_sphere;
    explicit Sphere(double R_) : R(R_), bounding_sphere(detail::sphereBoundingSphere(R_)) {}
};

// An axis-aligned ellipsoid with semi-axis half-lengths (a, b, c) along the
// local x, y, z axes: { x : (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1 }.
// P (= diag(1/a^2, 1/b^2, 1/c^2), the matrix of x'Px <= 1) and its upper
// factor U (= diag(1/a, 1/b, 1/c)) are members: the SOC constraint
// is ||U R^T (p - r)|| <= alpha.
struct Ellipsoid {
    double a, b, c;
    Eigen::Matrix3d P;
    Eigen::Matrix3d U;
    const BoundingSphere bounding_sphere;
    Ellipsoid(double a_, double b_, double c_)
        : a(a_),
          b(b_),
          c(c_),
          P(Eigen::Vector3d(1.0 / (a_ * a_), 1.0 / (b_ * b_), 1.0 / (c_ * c_)).asDiagonal()),
          U(Eigen::Vector3d(1.0 / a_, 1.0 / b_, 1.0 / c_).asDiagonal()),
          bounding_sphere(detail::ellipsoidBoundingSphere(a_, b_, c_)) {}
};

template <int NH>
struct Polytope {
    Eigen::Matrix<double, NH, 3> A;
    Eigen::Matrix<double, NH, 1> b;
    const BoundingSphere bounding_sphere;
    Polytope(const Eigen::Matrix<double, NH, 3>& A_, const Eigen::Matrix<double, NH, 1>& b_)
        : A(A_), b(b_), bounding_sphere(detail::polytopeBoundingSphere<NH>(A_, b_)) {}
};

template <int NH>
struct Polygon {
    Eigen::Matrix<double, NH, 2> A;
    Eigen::Matrix<double, NH, 1> b;
    double R; // "cushion" radius
    const BoundingSphere bounding_sphere;
    Polygon(const Eigen::Matrix<double, NH, 2>& A_, const Eigen::Matrix<double, NH, 1>& b_, double R_)
        : A(A_), b(b_), R(R_), bounding_sphere(detail::polygonBoundingSphere<NH>(A_, b_, R_)) {}
};

// Half-space: plane normal n and point p with d = n·p. Always shape 1, and
// does not scale with alpha. The row normal·p <= d is flipped per query
// (applyPlaneFlip) to normal·p >= d when body 2's centre is on the -n side;
// the result then carries plane_flipped.
struct Plane {
    Eigen::Vector3d normal;
    double d;
    Plane() : normal(Eigen::Vector3d::UnitX()), d(1.0) {}
    Plane(const Eigen::Vector3d& normal_, const Eigen::Vector3d& point_)
        : normal(normal_.normalized()), d(normal_.normalized().dot(point_)) {}
};

// Strict convexity: a shape whose boundary contains no straight line
// segment. Used by contact_degeneracy.hpp/contact_manifold.hpp to skip the
// degeneracy/manifold computation whenever
// either touching shape qualifies: two convex bodies in contact, at least
// one strictly convex, can only touch at a single point, and the combined valid normal at
// that point is the intersection of both bodies' normal cones -- which
// collapses to the smooth body's own single ray regardless of how
// degenerate the OTHER body's normal cone is (even a sharp vertex), so
// normal_cone_dim is provably 0 too.
//
// Only Sphere and Ellipsoid qualify among these 7 shapes. 
template <typename Shape>
struct IsStrictlyConvex : std::false_type {};
template <>
struct IsStrictlyConvex<Sphere> : std::true_type {};
template <>
struct IsStrictlyConvex<Ellipsoid> : std::true_type {};

// Unbounded half-space; must be shape 1. Only Plane.
template <typename Shape>
struct IsHalfspace : std::false_type {};
template <>
struct IsHalfspace<Plane> : std::true_type {};

} // namespace dcolpp::socp
