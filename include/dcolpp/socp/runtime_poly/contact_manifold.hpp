#pragma once
// dcolpp::socp::runtime_poly -- ContactDegeneracy / ContactManifold for a pure-ORT
// (PolytopeX vs PolytopeX, or Plane vs PolytopeX) contact: a shared face/edge
// under-represented by one witness point. Runtime-row-count copy of
// contact_degeneracy.hpp / contact_manifold.hpp with the SOC branches
// dropped (nx = 4). A hull-vs-curved-primitive manifold (SOC clip) is a
// follow-up; those pairs are point contacts unless the primitive has a flat
// face touching a hull face.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"

namespace dcolpp::socp::runtime_poly {

struct ContactDegeneracyX {
    int contact_manifold_dim = 0;
    int normal_cone_dim = 0;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
};

// A = G^T diag(z./s) G  (4x4).
inline Eigen::Matrix4d sensitivityA(const StackVecX& s, const StackVecX& z, const ConstraintMatX<4>& G) {
    const StackVecX sz = z.cwiseQuotient(s);
    return G.transpose() * (sz.asDiagonal() * G);
}

// normalConeRank (contact_degeneracy.hpp), ORT-only: active rows of G, rank.
inline std::pair<int, int> normalConeRank(const StackVecX& z, const ConstraintMatX<4>& G, double kActiveTol,
                                          double kRelZeroTol) {
    const int n = static_cast<int>(z.size());
    int m = 0;
    for (int i = 0; i < n; ++i)
        if (z(i) > kActiveTol) ++m;

    Eigen::Matrix<double, Eigen::Dynamic, 4, 0, kMaxOrtRows, 4> A_active(m, 4);
    int mi = 0;
    for (int i = 0; i < n; ++i)
        if (z(i) > kActiveTol) A_active.row(mi++) = G.row(i);

    Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, 4, 0, kMaxOrtRows, 4>> lu(A_active);
    lu.setThreshold(kRelZeroTol);
    return {m, static_cast<int>(lu.rank())};
}

inline ContactDegeneracyX contactDegeneracy(const StackVecX& s, const StackVecX& z, const ConstraintMatX<4>& G) {
    constexpr double kActiveTol = 1e-3, kRelZeroTol = 1e-3;
    const Eigen::Matrix4d A = sensitivityA(s, z, G);
    const double g_scale = std::max(G.squaredNorm(), 1.0);

    Eigen::FullPivLU<Eigen::Matrix4d> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = 4 - static_cast<int>(luA.rank());

    const auto [m, rAct] = normalConeRank(z, G, kActiveTol, kRelZeroTol);
    const int d_d = m - rAct;

    ContactDegeneracyX out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = d_d;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (d_d == 0);
    return out;
}

struct ContactManifoldX {
    std::vector<Eigen::Vector3d> witness_points;
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
};

inline ContactManifoldX contactManifold(const Eigen::Vector4d& x, const StackVecX& s, const StackVecX& z,
                                        const ConstraintMatX<4>& G, int K = 4) {
    constexpr double kTwoPi = 6.28318530717958647692;
    constexpr double kActiveTol = 1e-3, kRelZeroTol = 1e-3;
    const int n = static_cast<int>(s.size());

    const Eigen::Matrix4d A = sensitivityA(s, z, G);
    const double g_scale = std::max(G.squaredNorm(), 1.0);
    Eigen::FullPivLU<Eigen::Matrix4d> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = 4 - static_cast<int>(luA.rank());

    const auto [m, rAct] = normalConeRank(z, G, kActiveTol, kRelZeroTol);
    const int d_d = m - rAct;

    ContactManifoldX out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = d_d;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (d_d == 0);

    if (d_p == 0) {
        out.witness_points.push_back(x.head<3>());
        return out;
    }

    Eigen::JacobiSVD<Eigen::Matrix4d> svdA(A, Eigen::ComputeFullV);

    auto clip = [&](const StackVecX& s_here, const Eigen::Vector4d& dir) -> std::pair<double, double> {
        double t_min = -1e18, t_max = 1e18;
        for (int i = 0; i < n; ++i) {
            const double Gid = G.row(i).dot(dir);
            if (std::abs(Gid) < 1e-14) continue;
            const double t_bound = s_here(i) / Gid;
            if (Gid > 0) t_max = std::min(t_max, t_bound);
            else t_min = std::max(t_min, t_bound);
        }
        return {t_min, t_max};
    };
    auto shiftSlack = [&](const StackVecX& s_here, const Eigen::Vector4d& dir, double t) -> StackVecX {
        return StackVecX(s_here - t * (G * dir));
    };

    const Eigen::Vector4d d1 = svdA.matrixV().col(3);
    if (d_p == 1) {
        const auto [t_min, t_max] = clip(s, d1);
        out.witness_points.push_back(Eigen::Vector4d(x + t_min * d1).head<3>());
        out.witness_points.push_back(Eigen::Vector4d(x + t_max * d1).head<3>());
        return out;
    }

    const Eigen::Vector4d d2 = svdA.matrixV().col(2);

    const auto [t1min, t1max] = clip(s, d1);
    const double t1c = 0.5 * (t1min + t1max);
    const Eigen::Vector4d c1 = x + t1c * d1;
    const StackVecX s1 = shiftSlack(s, d1, t1c);

    const auto [t2min, t2max] = clip(s1, d2);
    const double t2c = 0.5 * (t2min + t2max);
    const Eigen::Vector4d c2 = c1 + t2c * d2;
    const StackVecX s2 = shiftSlack(s1, d2, t2c);

    K = std::max(K, 3);
    const int M = std::max(2 * K, 8);
    std::vector<Eigen::Vector3d> candidates;
    candidates.reserve(M);
    for (int mi = 0; mi < M; ++mi) {
        const double theta = kTwoPi * mi / M;
        const Eigen::Vector4d dir = std::cos(theta) * d1 + std::sin(theta) * d2;
        const auto [tm_min, tm_max] = clip(s2, dir);
        (void)tm_min;
        candidates.push_back(Eigen::Vector4d(c2 + tm_max * dir).head<3>());
    }

    std::vector<bool> used(candidates.size(), false);
    int best = 0;
    double best_d = -1.0;
    const Eigen::Vector3d center3 = c2.head<3>();
    for (int mi = 0; mi < static_cast<int>(candidates.size()); ++mi) {
        const double d = (candidates[mi] - center3).squaredNorm();
        if (d > best_d) { best_d = d; best = mi; }
    }
    used[best] = true;
    out.witness_points.push_back(candidates[best]);
    while (static_cast<int>(out.witness_points.size()) < K) {
        int next = -1;
        double next_d = -1.0;
        for (int mi = 0; mi < static_cast<int>(candidates.size()); ++mi) {
            if (used[mi]) continue;
            double min_d = 1e300;
            for (const auto& p : out.witness_points) min_d = std::min(min_d, (candidates[mi] - p).squaredNorm());
            if (min_d > next_d) { next_d = min_d; next = mi; }
        }
        if (next < 0) break;
        used[next] = true;
        out.witness_points.push_back(candidates[next]);
    }
    return out;
}

} // namespace dcolpp::socp::runtime_poly
