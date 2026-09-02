#pragma once
// dcolpp::broadphase -- bounding-sphere brute broadphase.
//
// Cheapest cull: each shape stores its circumradius, and the world centre is
// just the transform translation. Overlap is a single squared-distance test.
// Best for ball-like bodies; looser than AABB but cheaper to build.

#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "dcolpp/socp/primitives.hpp"

namespace dcolpp::broadphase {

struct BoundSphere {
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    double r = 0.0;
};

inline bool overlaps(const BoundSphere& a, const BoundSphere& b) {
    const double rr = a.r + b.r;
    return (a.c - b.c).squaredNorm() <= rr * rr;
}

// World bounding sphere of a posed shape: { g.translation(), circumradius }.
// Rotation-invariant; nothing to compute.
template <class Shape>
BoundSphere worldBoundSphere(const Shape& s, const Eigen::Matrix4d& g) {
    return {g.template block<3, 1>(0, 3), s.bounding_sphere.outer};
}

// A half-space has no finite sphere -- keep Planes out of the broadphase.
inline BoundSphere worldBoundSphere(const socp::Plane&, const Eigen::Matrix4d&) {
    return {Eigen::Vector3d::Zero(), std::numeric_limits<double>::infinity()};
}

struct BruteSphereBroadphase {
    std::vector<BoundSphere> spheres; // index i is the caller's body id

    template <class F>
    void forEachOverlappingPair(F&& cb) const {
        const int n = static_cast<int>(spheres.size());
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (overlaps(spheres[i], spheres[j])) cb(i, j);
    }

    std::vector<std::pair<int, int>> overlappingPairs() const {
        std::vector<std::pair<int, int>> out;
        forEachOverlappingPair([&](int i, int j) { out.emplace_back(i, j); });
        return out;
    }

    template <class F>
    void query(const BoundSphere& q, F&& cb) const {
        const int n = static_cast<int>(spheres.size());
        for (int i = 0; i < n; ++i)
            if (overlaps(spheres[i], q)) cb(i);
    }
};

// Filtered form -- test only the `allowed` pairs (see brute.hpp).
template <class F>
void forEachActivePair(const std::vector<BoundSphere>& spheres,
                       const std::vector<std::pair<int, int>>& allowed, F&& cb) {
    for (const auto& p : allowed)
        if (overlaps(spheres[p.first], spheres[p.second])) cb(p.first, p.second);
}

} // namespace dcolpp::broadphase
