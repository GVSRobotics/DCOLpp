#pragma once
// dcolpp::broadphase -- brute-force broadphase (method "A").
//
// No data structure: just scan pairs of world AABBs. O(N^2) for the
// all-pairs form, O(M) for the filtered form. No incremental state --
// rebuild `boxes` every step.
//
// Best for small N. Simpler and cache-friendly than tree broadphase,
// but use the tree past a few hundred bodies.

#include <utility>
#include <vector>

#include "dcolpp/broadphase/aabb.hpp"

namespace dcolpp::broadphase {

struct BruteBroadphase {
    std::vector<Aabb> boxes; // index i is the caller's body id

    // Every pair whose boxes currently overlap, once, as cb(i, j) with i < j.
    template <class F>
    void forEachOverlappingPair(F&& cb) const {
        const int n = static_cast<int>(boxes.size());
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (overlaps(boxes[i], boxes[j])) cb(i, j);
    }

    std::vector<std::pair<int, int>> overlappingPairs() const {
        std::vector<std::pair<int, int>> out;
        forEachOverlappingPair([&](int i, int j) { out.emplace_back(i, j); });
        return out;
    }

    // Every box overlapping `b`, as cb(i).
    template <class F>
    void query(const Aabb& b, F&& cb) const {
        const int n = static_cast<int>(boxes.size());
        for (int i = 0; i < n; ++i)
            if (overlaps(boxes[i], b)) cb(i);
    }
};

// Filtered form -- the recommended path for a robot model. You already know
// which body pairs CAN collide (a collision filter / self-collision
// adjacency list). Test only those: cb(i, j) fires for the `allowed` pairs
// whose boxes currently overlap.
template <class F>
void forEachActivePair(const std::vector<Aabb>& boxes, const std::vector<std::pair<int, int>>& allowed,
                       F&& cb) {
    for (const auto& p : allowed)
        if (overlaps(boxes[p.first], boxes[p.second])) cb(p.first, p.second);
}

} // namespace dcolpp::broadphase
