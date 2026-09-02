#pragma once
// Dynamic bounding-volume hierarchy over fat AABBs.
// Box2D-inspired broadphase: each leaf carries skin to tolerate small motion
// without reinsertion, and insertion uses the surface-area heuristic with
// single-rotation balancing. Queries and overlapping-pair iteration are
// iterative DFS. Header-only, Eigen-only; use for large-N culling.

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "dcolpp/broadphase/aabb.hpp"

namespace dcolpp::broadphase {

class AabbTree {
public:
    static constexpr int kNil = -1;

    struct Config {
        // Leaf skin = fat_margin_rel * (box's largest extent) + fat_margin_abs.
        // Relative by default so it scales with body size; set _rel = 0 for a
        // fixed absolute skin.
        double fat_margin_rel = 0.05;
        double fat_margin_abs = 0.0;
        double motion_predict = 2.0; // extra skin along the displacement passed to update()
    };

    AabbTree() = default;
    explicit AabbTree(Config cfg) : cfg_(cfg) {}

    // Insert a leaf for the tight box `tight`; returns its proxy handle.
    int insert(const Aabb& tight, int id) {
        const int p = allocNode();
        nodes_[p].aabb = fatten(tight);
        nodes_[p].id = id;
        nodes_[p].height = 0;
        insertLeaf(p);
        ++leaf_count_;
        return p;
    }

    void remove(int proxy) {
        assert(proxy >= 0 && nodes_[proxy].isLeaf());
        removeLeaf(proxy);
        freeNode(proxy);
        --leaf_count_;
    }

    // Re-fit a moved proxy. No-op (returns false) while its tight box still
    // sits inside the fat one; otherwise rebuilds the fat box (skin + a slug
    // along `displacement`) and re-inserts.
    bool update(int proxy, const Aabb& tight, const Eigen::Vector3d& displacement = Eigen::Vector3d::Zero()) {
        assert(proxy >= 0 && nodes_[proxy].isLeaf());
        if (nodes_[proxy].aabb.contains(tight)) return false;
        Aabb fat = fatten(tight);
        const Eigen::Vector3d d = cfg_.motion_predict * displacement;
        for (int i = 0; i < 3; ++i) {
            if (d(i) < 0.0)
                fat.lo(i) += d(i);
            else
                fat.hi(i) += d(i);
        }
        removeLeaf(proxy);
        nodes_[proxy].aabb = fat;
        insertLeaf(proxy);
        return true;
    }

    void clear() {
        nodes_.clear();
        free_ = kNil;
        root_ = kNil;
        leaf_count_ = 0;
    }

    int id(int proxy) const { return nodes_[proxy].id; }
    const Aabb& fatAabb(int proxy) const { return nodes_[proxy].aabb; }
    int size() const { return leaf_count_; }
    int height() const { return root_ == kNil ? 0 : nodes_[root_].height; }

    // Every leaf whose fat box overlaps `box`; cb is called with the leaf id.
    template <class F>
    void query(const Aabb& box, F&& cb) const {
        queryProxy(box, [&](int p) { cb(nodes_[p].id); });
    }

    // Every unordered pair of leaves whose fat boxes overlap, once, as
    // cb(id_a, id_b).
    template <class F>
    void forEachOverlappingPair(F&& cb) const {
        for (int p = 0; p < static_cast<int>(nodes_.size()); ++p) {
            if (nodes_[p].height != 0) continue; // live leaves only
            queryProxy(nodes_[p].aabb, [&](int q) {
                if (q > p) cb(nodes_[p].id, nodes_[q].id); // each pair from its lower proxy
            });
        }
    }

    std::vector<std::pair<int, int>> overlappingPairs() const {
        std::vector<std::pair<int, int>> out;
        forEachOverlappingPair([&](int a, int b) { out.emplace_back(a, b); });
        return out;
    }

    // Structural check for tests: parent links, heights, and every internal
    // box exactly the merge of its children.
    bool validate() const {
        if (root_ == kNil) return leaf_count_ == 0;
        int leaves = 0;
        const bool ok = validateNode(root_, kNil, leaves);
        return ok && leaves == leaf_count_;
    }

private:
    struct Node {
        Aabb aabb;
        int id = -1;
        int parent = kNil;
        int child1 = kNil;
        int child2 = kNil;
        int height = -1; // -1 free, 0 leaf, >=1 internal
        int next = kNil; // free list
        bool isLeaf() const { return child1 == kNil; }
    };

    static constexpr int kStack = 256;

    Aabb fatten(const Aabb& tight) const {
        return tight.expanded(cfg_.fat_margin_rel * tight.extents().maxCoeff() + cfg_.fat_margin_abs);
    }

    int allocNode() {
        if (free_ == kNil) {
            const int old = static_cast<int>(nodes_.size());
            nodes_.resize(old ? old * 2 : 16);
            for (int i = old; i < static_cast<int>(nodes_.size()); ++i) {
                nodes_[i] = Node{};
                nodes_[i].next = (i + 1 < static_cast<int>(nodes_.size())) ? i + 1 : kNil;
            }
            free_ = old;
        }
        const int n = free_;
        free_ = nodes_[n].next;
        nodes_[n].parent = kNil;
        nodes_[n].child1 = kNil;
        nodes_[n].child2 = kNil;
        nodes_[n].height = 0;
        nodes_[n].next = kNil;
        return n;
    }

    void freeNode(int n) {
        nodes_[n].next = free_;
        nodes_[n].height = -1;
        free_ = n;
    }

    template <class F>
    void queryProxy(const Aabb& box, F&& cb) const {
        if (root_ == kNil) return;
        int stack[kStack];
        int sp = 0;
        stack[sp++] = root_;
        while (sp) {
            const int n = stack[--sp];
            const Node& nd = nodes_[n];
            if (!overlaps(nd.aabb, box)) continue;
            if (nd.isLeaf()) {
                cb(n);
            } else {
                assert(sp + 2 <= kStack);
                stack[sp++] = nd.child1;
                stack[sp++] = nd.child2;
            }
        }
    }

    void insertLeaf(int leaf) {
        if (root_ == kNil) {
            root_ = leaf;
            nodes_[leaf].parent = kNil;
            return;
        }

        // 1. descend to the best sibling by the surface-area heuristic.
        const Aabb leafBox = nodes_[leaf].aabb;
        int index = root_;
        while (!nodes_[index].isLeaf()) {
            const int c1 = nodes_[index].child1;
            const int c2 = nodes_[index].child2;
            const double area = nodes_[index].aabb.perimeter();
            const double combinedArea = merge(nodes_[index].aabb, leafBox).perimeter();
            const double costHere = 2.0 * combinedArea;
            const double inherit = 2.0 * (combinedArea - area);

            auto descend = [&](int child) {
                const double merged = merge(leafBox, nodes_[child].aabb).perimeter();
                const double base = nodes_[child].isLeaf() ? 0.0 : nodes_[child].aabb.perimeter();
                return merged - base + inherit;
            };
            const double cost1 = descend(c1);
            const double cost2 = descend(c2);

            if (costHere < cost1 && costHere < cost2) break;
            index = (cost1 < cost2) ? c1 : c2;
        }

        // 2. splice a new parent above that sibling.
        const int sibling = index;
        const int oldParent = nodes_[sibling].parent;
        const int newParent = allocNode();
        nodes_[newParent].parent = oldParent;
        nodes_[newParent].id = -1;
        nodes_[newParent].aabb = merge(leafBox, nodes_[sibling].aabb);
        nodes_[newParent].height = nodes_[sibling].height + 1;

        if (oldParent != kNil) {
            if (nodes_[oldParent].child1 == sibling)
                nodes_[oldParent].child1 = newParent;
            else
                nodes_[oldParent].child2 = newParent;
        } else {
            root_ = newParent;
        }
        nodes_[newParent].child1 = sibling;
        nodes_[newParent].child2 = leaf;
        nodes_[sibling].parent = newParent;
        nodes_[leaf].parent = newParent;

        // 3. walk back to the root, balancing and re-fitting.
        int i = nodes_[leaf].parent;
        while (i != kNil) {
            i = balance(i);
            const int c1 = nodes_[i].child1;
            const int c2 = nodes_[i].child2;
            nodes_[i].height = 1 + std::max(nodes_[c1].height, nodes_[c2].height);
            nodes_[i].aabb = merge(nodes_[c1].aabb, nodes_[c2].aabb);
            i = nodes_[i].parent;
        }
    }

    void removeLeaf(int leaf) {
        if (leaf == root_) {
            root_ = kNil;
            return;
        }
        const int parent = nodes_[leaf].parent;
        const int grand = nodes_[parent].parent;
        const int sibling = (nodes_[parent].child1 == leaf) ? nodes_[parent].child2 : nodes_[parent].child1;

        if (grand == kNil) {
            root_ = sibling;
            nodes_[sibling].parent = kNil;
            freeNode(parent);
            return;
        }
        if (nodes_[grand].child1 == parent)
            nodes_[grand].child1 = sibling;
        else
            nodes_[grand].child2 = sibling;
        nodes_[sibling].parent = grand;
        freeNode(parent);

        int i = grand;
        while (i != kNil) {
            i = balance(i);
            const int c1 = nodes_[i].child1;
            const int c2 = nodes_[i].child2;
            nodes_[i].aabb = merge(nodes_[c1].aabb, nodes_[c2].aabb);
            nodes_[i].height = 1 + std::max(nodes_[c1].height, nodes_[c2].height);
            i = nodes_[i].parent;
        }
    }

    // Single rotation of subtree iA if its two child subtrees differ in
    // height by more than one. Faithful port of b2DynamicTree::Balance.
    int balance(int iA) {
        Node& A = nodes_[iA];
        if (A.isLeaf() || A.height < 2) return iA;

        const int iB = A.child1;
        const int iC = A.child2;
        const int diff = nodes_[iC].height - nodes_[iB].height;

        if (diff > 1) { // rotate C up
            const int iF = nodes_[iC].child1;
            const int iG = nodes_[iC].child2;
            nodes_[iC].child1 = iA;
            nodes_[iC].parent = A.parent;
            A.parent = iC;
            reparent(iC, iA);
            if (nodes_[iF].height > nodes_[iG].height) {
                nodes_[iC].child2 = iF;
                A.child2 = iG;
                nodes_[iG].parent = iA;
                A.aabb = merge(nodes_[iB].aabb, nodes_[iG].aabb);
                nodes_[iC].aabb = merge(A.aabb, nodes_[iF].aabb);
                A.height = 1 + std::max(nodes_[iB].height, nodes_[iG].height);
                nodes_[iC].height = 1 + std::max(A.height, nodes_[iF].height);
            } else {
                nodes_[iC].child2 = iG;
                A.child2 = iF;
                nodes_[iF].parent = iA;
                A.aabb = merge(nodes_[iB].aabb, nodes_[iF].aabb);
                nodes_[iC].aabb = merge(A.aabb, nodes_[iG].aabb);
                A.height = 1 + std::max(nodes_[iB].height, nodes_[iF].height);
                nodes_[iC].height = 1 + std::max(A.height, nodes_[iG].height);
            }
            return iC;
        }
        if (diff < -1) { // rotate B up
            const int iD = nodes_[iB].child1;
            const int iE = nodes_[iB].child2;
            nodes_[iB].child1 = iA;
            nodes_[iB].parent = A.parent;
            A.parent = iB;
            reparent(iB, iA);
            if (nodes_[iD].height > nodes_[iE].height) {
                nodes_[iB].child2 = iD;
                A.child1 = iE;
                nodes_[iE].parent = iA;
                A.aabb = merge(nodes_[iC].aabb, nodes_[iE].aabb);
                nodes_[iB].aabb = merge(A.aabb, nodes_[iD].aabb);
                A.height = 1 + std::max(nodes_[iC].height, nodes_[iE].height);
                nodes_[iB].height = 1 + std::max(A.height, nodes_[iD].height);
            } else {
                nodes_[iB].child2 = iE;
                A.child1 = iD;
                nodes_[iD].parent = iA;
                A.aabb = merge(nodes_[iC].aabb, nodes_[iD].aabb);
                nodes_[iB].aabb = merge(A.aabb, nodes_[iE].aabb);
                A.height = 1 + std::max(nodes_[iC].height, nodes_[iD].height);
                nodes_[iB].height = 1 + std::max(A.height, nodes_[iE].height);
            }
            return iB;
        }
        return iA;
    }

    // `promoted` has just taken `old`'s slot; fix the link from their shared
    // (old) parent, which promoted->parent now points at.
    void reparent(int promoted, int old) {
        const int p = nodes_[promoted].parent;
        if (p == kNil) {
            root_ = promoted;
            return;
        }
        if (nodes_[p].child1 == old)
            nodes_[p].child1 = promoted;
        else
            nodes_[p].child2 = promoted;
    }

    bool validateNode(int n, int parent, int& leaves) const {
        if (n == kNil) return false;
        const Node& nd = nodes_[n];
        if (nd.parent != parent) return false;
        if (nd.isLeaf()) {
            ++leaves;
            return nd.height == 0 && nd.child2 == kNil;
        }
        const int c1 = nd.child1;
        const int c2 = nd.child2;
        if (nd.height != 1 + std::max(nodes_[c1].height, nodes_[c2].height)) return false;
        const Aabb m = merge(nodes_[c1].aabb, nodes_[c2].aabb);
        if ((m.lo.array() != nd.aabb.lo.array()).any() || (m.hi.array() != nd.aabb.hi.array()).any())
            return false;
        return validateNode(c1, n, leaves) && validateNode(c2, n, leaves);
    }

    Config cfg_;
    std::vector<Node> nodes_;
    int free_ = kNil;
    int root_ = kNil;
    int leaf_count_ = 0;
};

} // namespace dcolpp::broadphase
