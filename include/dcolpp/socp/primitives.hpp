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
#include <vector>

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

// Unique feasible vertices of { A x <= b } (x in R^3): the C(NH,3) plane
// triples where three faces meet, kept if the point satisfies every
// inequality. O(NH^3), run ONCE in the Polytope ctor and cached on the
// struct. Where >3 faces meet, the same point comes from several triples --
// it is stored only once.
template <int NH>
std::vector<Eigen::Vector3d> polytopeVertices(const Eigen::Matrix<double, NH, 3>& A,
                                              const Eigen::Matrix<double, NH, 1>& b) {
    std::vector<Eigen::Vector3d> vs;
    const double bscale = 1.0 + b.cwiseAbs().maxCoeff();
    const double feas_tol = 1e-9 * bscale;
    const double dup_tol2 = (1e-7 * bscale) * (1e-7 * bscale);
    for (int i = 0; i < NH; ++i) {
        for (int j = i + 1; j < NH; ++j) {
            for (int k = j + 1; k < NH; ++k) {
                // the point where faces i, j, k meet -- Cramer's rule via the
                // scalar triple product (det) and vector triple products.
                const Eigen::Vector3d ai = A.row(i).transpose();
                const Eigen::Vector3d aj = A.row(j).transpose();
                const Eigen::Vector3d ak = A.row(k).transpose();
                const Eigen::Vector3d aj_x_ak = aj.cross(ak);
                const double det = ai.dot(aj_x_ak);
                const double scale = ai.norm() * aj.norm() * ak.norm();
                if (scale == 0.0 || std::abs(det) < 1e-9 * scale) continue; // planes not independent
                const Eigen::Vector3d y =
                    (b(i) * aj_x_ak + b(j) * ak.cross(ai) + b(k) * ai.cross(aj)) / det;
                if (!((A * y).array() <= b.array() + feas_tol).all()) continue; // outside another face
                bool seen = false;
                for (const auto& v : vs)
                    if ((v - y).squaredNorm() <= dup_tol2) { seen = true; break; }
                if (!seen) vs.push_back(y);
            }
        }
    }
    return vs;
}

// A y <= b in the shape's local frame; local origin assumed interior.
// inner: exact insphere radius (nearest-face distance).
// outer: exact circumradius = farthest precomputed vertex.
// Falls back to sqrt(3)*farthest-face if no vertex found (degenerate case).
template <int NH>
BoundingSphere polytopeBoundingSphere(const Eigen::Matrix<double, NH, 3>& A, const Eigen::Matrix<double, NH, 1>& b,
                                      const std::vector<Eigen::Vector3d>& verts) {
    double inner = std::numeric_limits<double>::infinity();
    double maxface = 0.0;
    for (int i = 0; i < NH; ++i) {
        const double d = b(i) / A.row(i).norm();
        inner = std::min(inner, d);
        maxface = std::max(maxface, d);
    }
    double outer = 0.0;
    for (const auto& v : verts) outer = std::max(outer, v.norm());
    if (!(outer > 0.0)) outer = std::sqrt(3.0) * maxface;

    return {inner, outer};
}

// Unique feasible 2D vertices of { A u <= b } (u in R^2): the C(NH,2)
// edge-line intersections that satisfy every inequality. Done ONCE in the
// Polygon ctor. A corner shared by >2 edges is stored only once.
template <int NH>
std::vector<Eigen::Vector2d> polygonVertices(const Eigen::Matrix<double, NH, 2>& A,
                                             const Eigen::Matrix<double, NH, 1>& b) {
    std::vector<Eigen::Vector2d> vs;
    const double bscale = 1.0 + b.cwiseAbs().maxCoeff();
    const double feas_tol = 1e-9 * bscale;
    const double dup_tol2 = (1e-7 * bscale) * (1e-7 * bscale);
    for (int i = 0; i < NH; ++i) {
        for (int j = i + 1; j < NH; ++j) {
            // the point where edges i, j meet -- 2x2 Cramer's rule.
            const Eigen::Vector2d ai = A.row(i).transpose();
            const Eigen::Vector2d aj = A.row(j).transpose();
            const double det = ai.x() * aj.y() - ai.y() * aj.x();
            const double scale = ai.norm() * aj.norm();
            if (scale == 0.0 || std::abs(det) < 1e-9 * scale) continue; // parallel edges
            const Eigen::Vector2d u((b(i) * aj.y() - b(j) * ai.y()) / det,
                                    (b(j) * ai.x() - b(i) * aj.x()) / det);
            if (!((A * u).array() <= b.array() + feas_tol).all()) continue;
            bool seen = false;
            for (const auto& v : vs)
                if ((v - u).squaredNorm() <= dup_tol2) { seen = true; break; }
            if (!seen) vs.push_back(u);
        }
    }
    return vs;
}

// A 2D convex polygon (A,b) in-plane, puffed out by cushion radius R in 3D
// (see problemMatrices(Polygon)).
// inner: exact -- the ball of radius R about the local origin.
// outer: exact -- (farthest precomputed 2D vertex) + R. Falls back to
// sqrt(2)*farthest-edge + R only if no vertex is found (degenerate).
template <int NH>
BoundingSphere polygonBoundingSphere(const Eigen::Matrix<double, NH, 2>& A, const Eigen::Matrix<double, NH, 1>& b,
                                      double R, const std::vector<Eigen::Vector2d>& verts) {
    double maxface2d = 0.0;
    for (int i = 0; i < NH; ++i) maxface2d = std::max(maxface2d, b(i) / A.row(i).norm());

    double vmax = 0.0;
    for (const auto& u : verts) vmax = std::max(vmax, u.norm());
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
    std::vector<Eigen::Vector3d> vertices; // unique feasible vertices, local frame -- precomputed once
    const BoundingSphere bounding_sphere;
    Polytope(const Eigen::Matrix<double, NH, 3>& A_, const Eigen::Matrix<double, NH, 1>& b_)
        : A(A_),
          b(b_),
          vertices(detail::polytopeVertices<NH>(A_, b_)),
          bounding_sphere(detail::polytopeBoundingSphere<NH>(A_, b_, vertices)) {}
};

template <int NH>
struct Polygon {
    Eigen::Matrix<double, NH, 2> A;
    Eigen::Matrix<double, NH, 1> b;
    double R;                             // "cushion" radius
    std::vector<Eigen::Vector2d> vertices; // unique feasible 2D vertices, local frame -- precomputed once
    const BoundingSphere bounding_sphere;
    Polygon(const Eigen::Matrix<double, NH, 2>& A_, const Eigen::Matrix<double, NH, 1>& b_, double R_)
        : A(A_),
          b(b_),
          R(R_),
          vertices(detail::polygonVertices<NH>(A_, b_)),
          bounding_sphere(detail::polygonBoundingSphere<NH>(A_, b_, R_, vertices)) {}
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
