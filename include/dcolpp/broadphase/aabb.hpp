#pragma once
// dcolpp::broadphase -- world-space axis-aligned bounding boxes for the
// collision primitives.
//
// Broadphase only: an Aabb is a loose cull box, never touched by the
// narrowphase SOCP. worldAabb(shape, g) is the tightest axis-aligned box
// around the posed shape, built from that shape's exact support function
// h_S(d) = max_{x in S} d . x. A caller inflates the box with a margin
// before feeding it to the tree.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "dcolpp/socp/primitives.hpp"

namespace dcolpp::broadphase {

struct Aabb {
    Eigen::Vector3d lo{Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity())};
    Eigen::Vector3d hi{Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity())};

    bool valid() const { return (lo.array() <= hi.array()).all(); }
    Eigen::Vector3d center() const { return 0.5 * (lo + hi); }
    Eigen::Vector3d extents() const { return hi - lo; }

    // Surface area -- the SAH cost used by the tree. 0 for an empty box.
    double perimeter() const {
        if (!valid()) return 0.0;
        const Eigen::Vector3d d = hi - lo;
        return 2.0 * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
    }
    Aabb expanded(double m) const {
        Aabb o = *this;
        o.lo.array() -= m;
        o.hi.array() += m;
        return o;
    }
    bool contains(const Aabb& b) const {
        return (lo.array() <= b.lo.array()).all() && (b.hi.array() <= hi.array()).all();
    }
};

inline bool overlaps(const Aabb& a, const Aabb& b) {
    return (a.lo.array() <= b.hi.array()).all() && (b.lo.array() <= a.hi.array()).all();
}
inline Aabb merge(const Aabb& a, const Aabb& b) {
    Aabb o;
    o.lo = a.lo.cwiseMin(b.lo);
    o.hi = a.hi.cwiseMax(b.hi);
    return o;
}

namespace detail {

inline double radial(const Eigen::Vector3d& d) { return std::sqrt(d.y() * d.y() + d.z() * d.z()); }

// h_S(d) = max_{x in S} d . x, S in its own local frame. d need not be unit.
inline double supportLocal(const socp::Sphere& s, const Eigen::Vector3d& d) { return s.R * d.norm(); }

inline double supportLocal(const socp::Capsule& s, const Eigen::Vector3d& d) {
    return 0.5 * s.L * std::abs(d.x()) + s.R * d.norm();
}
inline double supportLocal(const socp::Cylinder& s, const Eigen::Vector3d& d) {
    return 0.5 * s.L * std::abs(d.x()) + s.R * radial(d);
}
inline double supportLocal(const socp::Ellipsoid& s, const Eigen::Vector3d& d) {
    return std::sqrt(s.a * s.a * d.x() * d.x() + s.b * s.b * d.y() * d.y() + s.c * s.c * d.z() * d.z());
}
// Solid cone = conv( apex at x=-3H/4, base circle radius H*tan(beta) at x=+H/4 ).
inline double supportLocal(const socp::Cone& s, const Eigen::Vector3d& d) {
    const double Rb = s.H * std::tan(s.beta);
    const double apex = -0.75 * s.H * d.x();
    const double rim = 0.25 * s.H * d.x() + Rb * radial(d);
    return std::max(apex, rim);
}
// Frustum = conv( bottom circle R_bottom at x=-L/2, top circle R_top at x=+L/2 ).
inline double supportLocal(const socp::TruncatedCone& s, const Eigen::Vector3d& d) {
    const double bottom = -0.5 * s.L * d.x() + s.R_bottom * radial(d);
    const double top = 0.5 * s.L * d.x() + s.R_top * radial(d);
    return std::max(bottom, top);
}
template <int NH>
double supportLocal(const socp::Polytope<NH>& s, const Eigen::Vector3d& d) {
    double best = -std::numeric_limits<double>::infinity();
    for (const auto& v : s.vertices) best = std::max(best, d.dot(v));
    return best;
}
// 2D polygon in the local z=0 plane, Minkowski-summed with a ball of radius R.
template <int NH>
double supportLocal(const socp::Polygon<NH>& s, const Eigen::Vector3d& d) {
    const Eigen::Vector2d d2(d.x(), d.y());
    double best = -std::numeric_limits<double>::infinity();
    for (const auto& u : s.vertices) best = std::max(best, d2.dot(u));
    return best + s.R * d.norm();
}

} // namespace detail

// Tightest world-space AABB of `shape` posed by g (4x4 homogeneous). Works
// for every bounded primitive; the two support evaluations per axis keep it
// tight for the asymmetric ones (Cone, TruncatedCone).
template <typename Shape>
Aabb worldAabb(const Shape& shape, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d Rt = g.template block<3, 3>(0, 0).transpose();
    const Eigen::Vector3d t = g.template block<3, 1>(0, 3);
    Aabb box;
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d ei = Rt.col(i); // R^T e_i, unit
        box.hi(i) = t(i) + detail::supportLocal(shape, Eigen::Vector3d(ei));
        box.lo(i) = t(i) - detail::supportLocal(shape, Eigen::Vector3d(-ei));
    }
    return box;
}

// Polytope / Polygon: just transform the vertices the ctor already cached
// (the generic path would call the support function once per axis).
template <int NH>
Aabb worldAabb(const socp::Polytope<NH>& s, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
    const Eigen::Vector3d t = g.block<3, 1>(0, 3);
    Aabb box;
    for (const auto& v : s.vertices) {
        const Eigen::Vector3d w = R * v + t;
        box.lo = box.lo.cwiseMin(w);
        box.hi = box.hi.cwiseMax(w);
    }
    return box;
}

template <int NH>
Aabb worldAabb(const socp::Polygon<NH>& s, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
    const Eigen::Vector3d t = g.block<3, 1>(0, 3);
    Aabb box;
    for (const auto& u : s.vertices) {
        const Eigen::Vector3d w = R * Eigen::Vector3d(u.x(), u.y(), 0.0) + t;
        box.lo = box.lo.cwiseMin(w);
        box.hi = box.hi.cwiseMax(w);
    }
    return box.expanded(s.R); // the cushion ball
}

// A half-space has no finite box -- keep Planes OUT of the tree and test
// them linearly against every other proxy. Provided only for completeness.
inline Aabb worldAabb(const socp::Plane&, const Eigen::Matrix4d&) {
    const double inf = std::numeric_limits<double>::infinity();
    Aabb box;
    box.lo.setConstant(-inf);
    box.hi.setConstant(inf);
    return box;
}

} // namespace dcolpp::broadphase
