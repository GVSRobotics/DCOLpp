// dcolpp::broadphase -- worldAabb() boxes, the brute BruteBroadphase, and the
// dynamic AabbTree.
//
//  * worldAabb: exact closed forms (Sphere, Ellipsoid, rotated cube) plus an
//    independent containment / tightness check that samples each primitive's
//    real surface -- no reuse of the support functions under test.
//  * AabbTree: overlappingPairs() and query() are checked against brute-force
//    O(N^2) over the tree's own fat boxes, through insert / update / remove
//    and a randomized stress loop, with validate() asserted throughout.
//  * BruteBroadphase: same pair set as the tree, plus the filtered form.
//  * end-to-end: a small scene, culled pairs vs. the true DCOL++ contact.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "portable_random.hpp"

#include "dcolpp/broadphase/aabb_tree.hpp"
#include "dcolpp/broadphase/brute.hpp"
#include "dcolpp/broadphase/sphere.hpp"
#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp"

using namespace dcolpp::broadphase;
using dcolpp::socp::Capsule;
using dcolpp::socp::Cone;
using dcolpp::socp::Cylinder;
using dcolpp::socp::Ellipsoid;
using dcolpp::socp::Polygon;
using dcolpp::socp::Polytope;
using dcolpp::socp::Sphere;
using dcolpp::socp::TruncatedCone;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

// portable [0,1) from mt19937 raw bits (same formula as portable_random.hpp)
double u01(std::mt19937& rng) {
    const std::uint64_t a = static_cast<std::uint32_t>(rng()) >> 5;
    const std::uint64_t b = static_cast<std::uint32_t>(rng()) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}
double uni(std::mt19937& rng, double lo, double hi) { return lo + (hi - lo) * u01(rng); }

Vector3d randUnit(std::mt19937& rng) {
    dcolpp_test::PortableNormal nd;
    Vector3d v(nd(rng), nd(rng), nd(rng));
    while (v.squaredNorm() < 1e-12) v = Vector3d(nd(rng), nd(rng), nd(rng));
    return v.normalized();
}

Matrix4d randomPose(std::mt19937& rng, double trange = 2.0) {
    dcolpp_test::PortableNormal nd;
    Vector6d xi;
    xi.head<3>() = 0.8 * Vector3d(nd(rng), nd(rng), nd(rng)); // rotation
    xi.tail<3>().setZero();
    Matrix4d g = dcolpp::se3::Exp(xi);
    g.block<3, 1>(0, 3) = trange * Vector3d(nd(rng), nd(rng), nd(rng));
    return g;
}

struct Box {
    Vector3d lo = Vector3d::Constant(1e300);
    Vector3d hi = Vector3d::Constant(-1e300);
    void add(const Vector3d& p) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
};

template <class SampleFn>
void checkAabb(const Aabb& box, const Matrix4d& g, SampleFn&& sample, int n, double tol, const char* what) {
    const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
    const Vector3d t = g.block<3, 1>(0, 3);
    Box s;
    bool all_inside = true;
    for (int i = 0; i < n; ++i) {
        const Vector3d w = R * sample() + t;
        s.add(w);
        all_inside &= (w.array() >= box.lo.array() - 1e-9).all() && (w.array() <= box.hi.array() + 1e-9).all();
    }
    INFO(what << ": every sampled surface point lies inside the box");
    REQUIRE(all_inside);
    INFO(what << ": box hugs the sampled extent (tol " << tol << ")");
    REQUIRE((box.hi.array() >= s.hi.array() - 1e-9).all());
    REQUIRE((box.lo.array() <= s.lo.array() + 1e-9).all());
    REQUIRE((box.hi - s.hi).cwiseAbs().maxCoeff() <= tol);
    REQUIRE((box.lo - s.lo).cwiseAbs().maxCoeff() <= tol);
}

Polytope<6> unitCube(double h = 0.5) {
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    return Polytope<6>(A, Eigen::Matrix<double, 6, 1>::Constant(h));
}

std::set<std::pair<int, int>> brutePairs(const AabbTree& tree, const std::vector<int>& proxies) {
    std::set<std::pair<int, int>> out;
    for (size_t i = 0; i < proxies.size(); ++i)
        for (size_t j = i + 1; j < proxies.size(); ++j)
            if (overlaps(tree.fatAabb(proxies[i]), tree.fatAabb(proxies[j])))
                out.insert({tree.id(proxies[i]), tree.id(proxies[j])});
    return out;
}

std::set<std::pair<int, int>> toSet(std::vector<std::pair<int, int>> v) {
    for (auto& p : v)
        if (p.first > p.second) std::swap(p.first, p.second);
    return {v.begin(), v.end()};
}

Aabb randBox(std::mt19937& rng, double spread) {
    const Vector3d c(uni(rng, -spread, spread), uni(rng, -spread, spread), uni(rng, -spread, spread));
    const Vector3d h(uni(rng, 0.1, 0.6), uni(rng, 0.1, 0.6), uni(rng, 0.1, 0.6));
    Aabb b;
    b.lo = c - h;
    b.hi = c + h;
    return b;
}

} // namespace

TEST_CASE("worldAabb: exact closed forms", "[broadphase][aabb]") {
    std::mt19937 rng(20260901u);

    SECTION("sphere is rotation-invariant, box = centre +- R") {
        const Sphere s(0.73);
        for (int t = 0; t < 40; ++t) {
            const Matrix4d g = randomPose(rng);
            const Aabb box = worldAabb(s, g);
            const Vector3d c = g.block<3, 1>(0, 3);
            REQUIRE((box.lo - (c.array() - 0.73).matrix()).norm() < 1e-12);
            REQUIRE((box.hi - (c.array() + 0.73).matrix()).norm() < 1e-12);
        }
    }

    SECTION("ellipsoid matches sqrt(sum_j (R_ij s_j)^2)") {
        const Vector3d s(0.4, 1.1, 0.7);
        const Ellipsoid e(s.x(), s.y(), s.z());
        for (int t = 0; t < 40; ++t) {
            const Matrix4d g = randomPose(rng);
            const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
            const Vector3d c = g.block<3, 1>(0, 3);
            const Aabb box = worldAabb(e, g);
            for (int i = 0; i < 3; ++i) {
                const double ext =
                    std::sqrt((R.row(i).transpose().array().square() * s.array().square()).sum());
                REQUIRE(std::abs(box.hi(i) - (c(i) + ext)) < 1e-12);
                REQUIRE(std::abs(box.lo(i) - (c(i) - ext)) < 1e-12);
            }
        }
    }

    SECTION("unit cube rotated 45 deg about z") {
        Vector6d xi;
        xi << 0, 0, M_PI / 4, 0, 0, 0;
        const Aabb box = worldAabb(unitCube(0.5), dcolpp::se3::Exp(xi));
        REQUIRE(std::abs(box.hi.x() - 0.5 * std::sqrt(2.0)) < 1e-12);
        REQUIRE(std::abs(box.hi.y() - 0.5 * std::sqrt(2.0)) < 1e-12);
        REQUIRE(std::abs(box.hi.z() - 0.5) < 1e-12);
    }

    SECTION("polytope vertices are de-duplicated (octahedron: 4 faces per vertex)") {
        Eigen::Matrix<double, 8, 3> A;
        int r = 0;
        for (int sx : {-1, 1})
            for (int sy : {-1, 1})
                for (int sz : {-1, 1}) A.row(r++) = Eigen::RowVector3d(sx, sy, sz).normalized();
        const Polytope<8> oct(A, Eigen::Matrix<double, 8, 1>::Constant(1.0 / std::sqrt(3.0)));
        REQUIRE(oct.vertices.size() == 6); // not 24 (= 6 vertices x C(4,3) triples)
        const Aabb box = worldAabb(oct, Matrix4d::Identity());
        REQUIRE((box.lo + Vector3d::Ones()).norm() < 1e-12);
        REQUIRE((box.hi - Vector3d::Ones()).norm() < 1e-12);
    }

    SECTION("polygon = transformed in-plane hull, expanded by the cushion R") {
        Eigen::Matrix<double, 4, 2> A;
        A << 1, 0, -1, 0, 0, 1, 0, -1;
        const Polygon<4> poly(A, Eigen::Vector4d(0.5, 0.5, 0.3, 0.3), 0.2);
        const auto& verts = poly.vertices;
        REQUIRE(verts.size() == 4);
        for (int t = 0; t < 40; ++t) {
            const Matrix4d g = randomPose(rng);
            const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
            const Vector3d c = g.block<3, 1>(0, 3);
            Box hull;
            for (const auto& v : verts) hull.add(R * Vector3d(v.x(), v.y(), 0.0) + c);
            const Aabb box = worldAabb(poly, g);
            REQUIRE((box.lo - (hull.lo.array() - 0.2).matrix()).norm() < 1e-12);
            REQUIRE((box.hi - (hull.hi.array() + 0.2).matrix()).norm() < 1e-12);
        }
    }
}

TEST_CASE("worldAabb: containment + tightness by surface sampling", "[broadphase][aabb]") {
    std::mt19937 rng(11111u);
    const int N = 40000;
    // Loose "not grossly oversized" bound: a curved extreme is a
    // measure-zero surface point, so the sampled max creeps up to it slowly.
    // Exact tightness is pinned by the closed-form section above; the
    // safety-critical direction (box CONTAINS the shape) is checked to 1e-9
    // on every sample.
    const double CURVED = 1.5e-2;

    for (int trial = 0; trial < 6; ++trial) {
        const Matrix4d g = randomPose(rng);

        {
            const Sphere s(0.6); // exact box is covered above; here just sample-tightness
            checkAabb(worldAabb(s, g), g, [&] { return Vector3d(0.6 * randUnit(rng)); }, N, CURVED, "sphere");
        }
        {
            const Capsule c(0.25, 1.4);
            checkAabb(
                worldAabb(c, g), g,
                [&] { return Vector3d(Vector3d(uni(rng, -0.7, 0.7), 0, 0) + 0.25 * randUnit(rng)); }, N, CURVED,
                "capsule");
        }
        {
            const Cylinder c(0.35, 1.2);
            checkAabb(
                worldAabb(c, g), g,
                [&] {
                    const double th = uni(rng, 0, 2 * M_PI);
                    if (u01(rng) < 0.5)
                        return Vector3d(uni(rng, -0.6, 0.6), 0.35 * std::cos(th), 0.35 * std::sin(th));
                    const double rho = 0.35 * std::sqrt(u01(rng));
                    return Vector3d(u01(rng) < 0.5 ? -0.6 : 0.6, rho * std::cos(th), rho * std::sin(th));
                },
                N, CURVED, "cylinder");
        }
        {
            const double H = 1.5, beta = 0.45;
            const Cone cone(H, beta);
            const double Rb = H * std::tan(beta);
            checkAabb(
                worldAabb(cone, g), g,
                [&] {
                    const double th = uni(rng, 0, 2 * M_PI);
                    if (u01(rng) < 0.15) return Vector3d(-0.75 * H, 0, 0);
                    if (u01(rng) < 0.5) {
                        const double x = uni(rng, -0.75 * H, 0.25 * H);
                        const double r = (x + 0.75 * H) * std::tan(beta);
                        return Vector3d(x, r * std::cos(th), r * std::sin(th));
                    }
                    const double rho = Rb * std::sqrt(u01(rng));
                    return Vector3d(0.25 * H, rho * std::cos(th), rho * std::sin(th));
                },
                N, CURVED, "cone");
        }
        {
            const double Rb = 0.8, Rt = 0.3, L = 1.1;
            const TruncatedCone tc(Rb, Rt, L);
            checkAabb(
                worldAabb(tc, g), g,
                [&] {
                    const double th = uni(rng, 0, 2 * M_PI);
                    const double x = uni(rng, -L / 2, L / 2);
                    const double r = Rb + (x + L / 2) / L * (Rt - Rb);
                    if (u01(rng) < 0.5) return Vector3d(x, r * std::cos(th), r * std::sin(th));
                    const bool topcap = u01(rng) < 0.5;
                    const double rho = (topcap ? Rt : Rb) * std::sqrt(u01(rng));
                    return Vector3d(topcap ? L / 2 : -L / 2, rho * std::cos(th), rho * std::sin(th));
                },
                N, CURVED, "truncated cone");
        }
        {
            const Ellipsoid e(0.5, 0.9, 1.3);
            checkAabb(
                worldAabb(e, g), g,
                [&] {
                    const double th = uni(rng, 0, 2 * M_PI), ph = std::acos(uni(rng, -1, 1));
                    return Vector3d(0.5 * std::sin(ph) * std::cos(th), 0.9 * std::sin(ph) * std::sin(th),
                                    1.3 * std::cos(ph));
                },
                N, CURVED, "ellipsoid");
        }
        {
            Eigen::Matrix<double, 6, 3> A;
            A << 1, 0.2, 0, -1, 0.1, 0, 0, 1, 0, 0.1, -1, 0.3, 0, 0.2, 1, 0, 0, -1;
            const Polytope<6> poly(A, (Eigen::Matrix<double, 6, 1>() << 0.6, 0.5, 0.7, 0.5, 0.8, 0.6).finished());
            const auto& verts = poly.vertices;
            REQUIRE(verts.size() >= 4);
            size_t vi = 0;
            checkAabb(
                worldAabb(poly, g), g, [&] { return verts[(vi++) % verts.size()]; },
                static_cast<int>(verts.size()), 1e-9, "polytope");
        }
    }
}

TEST_CASE("AabbTree: pairs and queries match brute force", "[broadphase][tree]") {
    std::mt19937 rng(777u);

    SECTION("insert / update / remove, pairs stay exact") {
        AabbTree tree(AabbTree::Config{0.05, 0.0, 2.0});
        const int N = 160;
        std::vector<int> proxies;
        std::vector<Aabb> tight(N);
        for (int i = 0; i < N; ++i) {
            tight[i] = randBox(rng, 3.0);
            proxies.push_back(tree.insert(tight[i], i));
        }
        REQUIRE(tree.validate());
        REQUIRE(tree.size() == N);
        REQUIRE(toSet(tree.overlappingPairs()) == brutePairs(tree, proxies));

        for (int i = 0; i < N; i += 2) {
            const Vector3d disp(uni(rng, -1.5, 1.5), uni(rng, -1.5, 1.5), uni(rng, -1.5, 1.5));
            tight[i].lo += disp;
            tight[i].hi += disp;
            tree.update(proxies[i], tight[i], disp);
        }
        REQUIRE(tree.validate());
        REQUIRE(toSet(tree.overlappingPairs()) == brutePairs(tree, proxies));

        for (int q = 0; q < 200; ++q) {
            const Aabb qb = randBox(rng, 3.5);
            std::set<int> got;
            tree.query(qb, [&](int id) { got.insert(id); });
            std::set<int> want;
            for (int i = 0; i < N; ++i)
                if (overlaps(tree.fatAabb(proxies[i]), qb)) want.insert(i);
            REQUIRE(got == want);
        }

        std::vector<int> live;
        for (int i = 0; i < N; ++i) {
            if (i % 3 == 0)
                tree.remove(proxies[i]);
            else
                live.push_back(proxies[i]);
        }
        REQUIRE(tree.validate());
        REQUIRE(tree.size() == static_cast<int>(live.size()));
        REQUIRE(toSet(tree.overlappingPairs()) == brutePairs(tree, live));
    }

    SECTION("randomized stress: 15 rounds of moves and churn") {
        AabbTree tree(AabbTree::Config{0.08, 0.0, 2.0});
        const int N = 220;
        std::vector<int> proxies(N);
        std::vector<Aabb> tight(N);
        std::vector<bool> alive(N, true);
        for (int i = 0; i < N; ++i) {
            tight[i] = randBox(rng, 4.0);
            proxies[i] = tree.insert(tight[i], i);
        }

        for (int round = 0; round < 15; ++round) {
            for (int i = 0; i < N; ++i) {
                const double roll = u01(rng);
                if (alive[i] && roll < 0.15) {
                    tree.remove(proxies[i]);
                    alive[i] = false;
                } else if (!alive[i] && roll < 0.5) {
                    tight[i] = randBox(rng, 4.0);
                    proxies[i] = tree.insert(tight[i], i);
                    alive[i] = true;
                } else if (alive[i] && roll < 0.7) {
                    const Vector3d disp(uni(rng, -1, 1), uni(rng, -1, 1), uni(rng, -1, 1));
                    tight[i].lo += disp;
                    tight[i].hi += disp;
                    tree.update(proxies[i], tight[i], disp);
                }
            }
            REQUIRE(tree.validate());

            std::vector<int> live;
            for (int i = 0; i < N; ++i)
                if (alive[i]) live.push_back(proxies[i]);
            REQUIRE(toSet(tree.overlappingPairs()) == brutePairs(tree, live));
            REQUIRE(tree.height() <=
                    4 * static_cast<int>(std::ceil(std::log2(std::max(2, tree.size())))) + 6);
        }
    }

    SECTION("degenerate: empty tree, single leaf") {
        AabbTree tree;
        REQUIRE(tree.validate());
        REQUIRE(tree.overlappingPairs().empty());
        Aabb b;
        b.lo = Vector3d(-1, -1, -1);
        b.hi = Vector3d(1, 1, 1);
        const int p = tree.insert(b, 42);
        REQUIRE(tree.validate());
        REQUIRE(tree.size() == 1);
        REQUIRE(tree.overlappingPairs().empty());
        int hits = 0;
        tree.query(b, [&](int id) {
            REQUIRE(id == 42);
            ++hits;
        });
        REQUIRE(hits == 1);
        tree.remove(p);
        REQUIRE(tree.validate());
        REQUIRE(tree.size() == 0);
    }
}

TEST_CASE("BruteBroadphase: same pairs as the tree, plus the filtered form", "[broadphase][brute]") {
    std::mt19937 rng(4242u);
    const int N = 120;

    BruteBroadphase bf;
    AabbTree tree(AabbTree::Config{0.0, 0.0, 0.0}); // no skin -> tree fat box == tight box
    std::vector<int> proxies(N);
    for (int i = 0; i < N; ++i) {
        const Aabb b = randBox(rng, 3.0);
        bf.boxes.push_back(b);
        proxies[i] = tree.insert(b, i);
    }

    // reference: hand O(N^2)
    std::set<std::pair<int, int>> ref;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (overlaps(bf.boxes[i], bf.boxes[j])) ref.insert({i, j});

    REQUIRE(toSet(bf.overlappingPairs()) == ref);
    REQUIRE(toSet(tree.overlappingPairs()) == ref); // A and B agree exactly

    for (int q = 0; q < 60; ++q) {
        const Aabb qb = randBox(rng, 3.5);
        std::set<int> got, want;
        bf.query(qb, [&](int i) { got.insert(i); });
        for (int i = 0; i < N; ++i)
            if (overlaps(bf.boxes[i], qb)) want.insert(i);
        REQUIRE(got == want);
    }

    // filtered form: only the allowed pairs, and only if their boxes overlap
    std::vector<std::pair<int, int>> allowed;
    for (auto& p : ref) allowed.push_back(p); // give it every truly-overlapping pair...
    allowed.push_back({0, 1});                // ...plus a couple that may or may not
    allowed.push_back({2, 3});
    std::set<std::pair<int, int>> active;
    forEachActivePair(bf.boxes, allowed, [&](int i, int j) { active.insert({i, j}); });
    for (auto& p : allowed)
        REQUIRE(active.count(p) == (overlaps(bf.boxes[p.first], bf.boxes[p.second]) ? 1u : 0u));
}

TEST_CASE("BruteSphereBroadphase: bounding-sphere cull", "[broadphase][sphere]") {
    std::mt19937 rng(9090u);
    const Sphere sph(0.5);
    const Capsule cap(0.2, 1.0);
    const Cylinder cyl(0.3, 0.8);
    const Polytope<6> box = unitCube(0.5);

    SECTION("worldBoundSphere = { g.translation, shape circumradius }") {
        const Matrix4d g = randomPose(rng);
        const auto bs = worldBoundSphere(cap, g);
        REQUIRE((bs.c - g.block<3, 1>(0, 3)).norm() < 1e-12);
        REQUIRE(bs.r == cap.bounding_sphere.outer);
    }

    SECTION("pairs match a hand O(N^2) sphere-overlap scan; contacts are never culled") {
        const int N = 40;
        BruteSphereBroadphase bf;
        std::vector<Matrix4d> gs(N);
        for (int i = 0; i < N; ++i) {
            gs[i] = randomPose(rng, 2.5);
            bf.spheres.push_back((i % 2) ? worldBoundSphere(cap, gs[i]) : worldBoundSphere(sph, gs[i]));
        }
        std::set<std::pair<int, int>> ref;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                if (overlaps(bf.spheres[i], bf.spheres[j])) ref.insert({i, j});
        REQUIRE(toSet(bf.overlappingPairs()) == ref);

        // conservative: a real (penetrating) contact must survive the cull
        Matrix4d ga = Matrix4d::Identity();
        Matrix4d gb = Matrix4d::Identity();
        gb(0, 3) = 0.6; // two r=0.3 cylinders, centres 0.6 apart -> penetrating
        BruteSphereBroadphase c;
        c.spheres = {worldBoundSphere(cyl, ga), worldBoundSphere(cyl, gb), worldBoundSphere(box, Matrix4d::Identity())};
        // shift the box far away
        Matrix4d gfar = Matrix4d::Identity();
        gfar(1, 3) = 20.0;
        c.spheres[2] = worldBoundSphere(box, gfar);
        const auto pairs = toSet(c.overlappingPairs());
        REQUIRE(dcolpp::socp::proximityContact(cyl, cyl, Matrix4d(ga.inverse() * gb)).gap < 0.0);
        REQUIRE(pairs.count({0, 1}) == 1); // penetrating pair kept
        REQUIRE(pairs.count({0, 2}) == 0); // far box culled
    }

    SECTION("filtered form respects the allow-list") {
        BruteSphereBroadphase bf;
        Matrix4d g0 = Matrix4d::Identity(), g1 = Matrix4d::Identity(), g2 = Matrix4d::Identity();
        g1(0, 3) = 0.7;    // overlaps g0
        g2(0, 3) = 10.0;   // overlaps nobody
        bf.spheres = {worldBoundSphere(sph, g0), worldBoundSphere(sph, g1), worldBoundSphere(sph, g2)};
        std::vector<std::pair<int, int>> allowed = {{0, 1}, {0, 2}};
        std::set<std::pair<int, int>> active;
        forEachActivePair(bf.spheres, allowed, [&](int i, int j) { active.insert({i, j}); });
        REQUIRE(active == std::set<std::pair<int, int>>{{0, 1}});
    }
}

TEST_CASE("broadphase end-to-end: no near pair is culled", "[broadphase][contact]") {
    const Sphere A(0.5);
    const Sphere B(0.5);
    const Polytope<6> C = unitCube(0.5);
    const Capsule D(0.3, 1.0);

    Matrix4d gA = Matrix4d::Identity();
    Matrix4d gB = Matrix4d::Identity();
    gB(0, 3) = 0.8; // radii 0.5 -> overlap
    Matrix4d gC = Matrix4d::Identity();
    gC(0, 3) = 0.8 + 0.5 + 0.5 + 0.05; // ~0.05 gap to B
    Matrix4d gD = Matrix4d::Identity();
    gD(1, 3) = 12.0;

    const double margin = 0.1;
    AabbTree tree(AabbTree::Config{0.0, margin, 0.0});
    tree.insert(worldAabb(A, gA), 0);
    tree.insert(worldAabb(B, gB), 1);
    tree.insert(worldAabb(C, gC), 2);
    tree.insert(worldAabb(D, gD), 3);

    const auto pairs = toSet(tree.overlappingPairs());

    REQUIRE(dcolpp::socp::proximityContact(A, B, Matrix4d(gA.inverse() * gB)).gap < 0.0);
    REQUIRE(dcolpp::socp::proximityContact(B, C, Matrix4d(gB.inverse() * gC)).gap < 2 * margin);

    REQUIRE(pairs.count({0, 1}) == 1);
    REQUIRE(pairs.count({1, 2}) == 1);
    REQUIRE(pairs.count({0, 2}) == 0);
    REQUIRE(pairs.count({0, 3}) == 0);
    REQUIRE(pairs.count({1, 3}) == 0);
    REQUIRE(pairs.count({2, 3}) == 0);
}
