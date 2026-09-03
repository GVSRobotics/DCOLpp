#pragma once
// dcolpp::socp::runtime_poly -- ContactManifold / ContactDegeneracy for PolytopeX
// (shape 1) vs a curved primitive (shape 2): a dynamic ORT block plus one
// fixed SOC block of size NSOC, decision width NX. Runtime-row-count copy of
// contact_manifold.hpp / contact_degeneracy.hpp; the SOC clip and the active-
// SOC-row test are the fixed-size versions.
//
// Relevant when a primitive's flat feature (a Cylinder/Cone/TruncatedCone cap,
// a Capsule shaft) lies against a hull face -> a line or patch contact.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "dcolpp/socp/cone_utils.hpp" // arrow
#include "dcolpp/socp/small_llt.hpp"
#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::arrow;

template <int NX, int NSOC>
struct ContactManifoldPrimResult {
    std::vector<Eigen::Vector3d> witness_points;
    int contact_manifold_dim = -1;
    int normal_cone_dim = -1;
    bool witness_jacobian_valid = true;
    bool normal_jacobian_valid = true;
};

// A = G^T (S^-1 Z) G, NX x NX. ORT: elementwise z./s. SOC: arrow(s)\arrow(z).
template <int NX, int NSOC>
Eigen::Matrix<double, NX, NX> sensitivityAPrim(const StackVecX& s, const StackVecX& z,
                                               const ConstraintMatX<NX>& G, int n_ort) {
    const int ns = n_ort + NSOC;
    Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX> SZG(ns, NX);
    SZG.topRows(n_ort) = (z.head(n_ort).cwiseQuotient(s.head(n_ort))).asDiagonal() * G.topRows(n_ort);
    if constexpr (NSOC > 0) {
        const Eigen::Matrix<double, NSOC, NSOC> Ssoc = arrow<NSOC>(Vec<NSOC>(s.template tail<NSOC>()));
        const Eigen::Matrix<double, NSOC, NSOC> Zsoc = arrow<NSOC>(Vec<NSOC>(z.template tail<NSOC>()));
        SmallLLT<NSOC> Sf;
        Sf.compute(Ssoc);
        SZG.template bottomRows<NSOC>() = Sf.solve(Zsoc) * G.template bottomRows<NSOC>();
    }
    return G.transpose() * SZG;
}

template <int NX, int NSOC>
std::pair<int, int> normalConeRankPrim(const StackVecX& z, const ConstraintMatX<NX>& G, int n_ort, double kActiveTol,
                                       double kRelZeroTol) {
    int m = 0;
    for (int i = 0; i < n_ort; ++i)
        if (z(i) > kActiveTol) ++m;
    bool soc_active = false;
    if constexpr (NSOC > 0) {
        soc_active = z.template tail<NSOC>().norm() > kActiveTol;
        if (soc_active) ++m;
    }
    Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX> Aa(m, NX);
    int mi = 0;
    for (int i = 0; i < n_ort; ++i)
        if (z(i) > kActiveTol) Aa.row(mi++) = G.row(i);
    if constexpr (NSOC > 0)
        if (soc_active) Aa.row(mi++) = z.template tail<NSOC>().transpose() * G.template bottomRows<NSOC>();

    Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX>> lu(Aa);
    lu.setThreshold(kRelZeroTol);
    return {m, static_cast<int>(lu.rank())};
}

// contact_manifold.hpp's ray clip, ORT loop runtime + the one SOC block.
template <int NX, int NSOC>
std::pair<double, double> clipRayPrim(const StackVecX& s_here, const Eigen::Matrix<double, NX, 1>& dir,
                                      const ConstraintMatX<NX>& G, int n_ort) {
    double t_min = -1e18, t_max = 1e18;
    for (int i = 0; i < n_ort; ++i) {
        const double Gid = G.row(i).dot(dir);
        if (std::abs(Gid) < 1e-14) continue;
        const double tb = s_here(i) / Gid;
        if (Gid > 0) t_max = std::min(t_max, tb);
        else t_min = std::max(t_min, tb);
    }
    if constexpr (NSOC > 0) {
        const Vec<NSOC> s0v = s_here.template tail<NSOC>();
        const Vec<NSOC> a = G.template bottomRows<NSOC>() * dir;
        if (a.norm() >= 1e-6 * std::max(s0v.norm(), 1.0)) {
            const double s0 = s0v(0), a0 = a(0);
            const auto s_tail = s0v.template tail<NSOC - 1>();
            const auto a_tail = a.template tail<NSOC - 1>();
            const double A_c = a0 * a0 - a_tail.squaredNorm();
            const double B_c = -2.0 * (s0 * a0 - s_tail.dot(a_tail));
            const double C_c = s0 * s0 - s_tail.squaredNorm();
            if (std::abs(A_c) < 1e-14) {
                if (std::abs(B_c) > 1e-14) {
                    const double tr = -C_c / B_c;
                    if (B_c > 0) t_max = std::min(t_max, tr);
                    else t_min = std::max(t_min, tr);
                }
            } else {
                const double disc = B_c * B_c - 4 * A_c * C_c;
                if (disc >= 0) {
                    const double sq = std::sqrt(disc);
                    double r1 = (-B_c - sq) / (2 * A_c), r2 = (-B_c + sq) / (2 * A_c);
                    if (r1 > r2) std::swap(r1, r2);
                    if (A_c < 0) {
                        t_min = std::max(t_min, r1);
                        t_max = std::min(t_max, r2);
                    }
                }
            }
        }
    }
    return {t_min, t_max};
}

template <int NX, int NSOC>
ContactManifoldPrimResult<NX, NSOC> contactDegeneracyPrim(const StackVecX& s, const StackVecX& z,
                                                         const ConstraintMatX<NX>& G, int n_ort) {
    constexpr double kActiveTol = 1e-3, kRelZeroTol = 1e-3;
    const Eigen::Matrix<double, NX, NX> A = sensitivityAPrim<NX, NSOC>(s, z, G, n_ort);
    const double g_scale = std::max(G.squaredNorm(), 1.0);
    Eigen::FullPivLU<Eigen::Matrix<double, NX, NX>> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = NX - static_cast<int>(luA.rank());
    const auto [m, rAct] = normalConeRankPrim<NX, NSOC>(z, G, n_ort, kActiveTol, kRelZeroTol);
    ContactManifoldPrimResult<NX, NSOC> out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = m - rAct;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (m - rAct == 0);
    return out;
}

template <int NX, int NSOC>
ContactManifoldPrimResult<NX, NSOC> contactManifoldPrim(const Eigen::Matrix<double, NX, 1>& x, const StackVecX& s,
                                                        const StackVecX& z, const ConstraintMatX<NX>& G, int n_ort,
                                                        int K = 4) {
    constexpr double kActiveTol = 1e-3, kRelZeroTol = 1e-3;

    const Eigen::Matrix<double, NX, NX> A = sensitivityAPrim<NX, NSOC>(s, z, G, n_ort);
    const double g_scale = std::max(G.squaredNorm(), 1.0);
    Eigen::FullPivLU<Eigen::Matrix<double, NX, NX>> luA(A);
    luA.setThreshold((kRelZeroTol * g_scale) / std::max(luA.maxPivot(), 1e-300));
    const int d_p = NX - static_cast<int>(luA.rank());

    const auto [m, rAct] = normalConeRankPrim<NX, NSOC>(z, G, n_ort, kActiveTol, kRelZeroTol);

    ContactManifoldPrimResult<NX, NSOC> out;
    out.contact_manifold_dim = d_p;
    out.normal_cone_dim = m - rAct;
    out.witness_jacobian_valid = (d_p == 0);
    out.normal_jacobian_valid = (m - rAct == 0);
    // d_p == 0: unique witness. d_p >= 2 with a curved primitive in the pair
    // (SOC block): the degenerate set is a genuine 2D+ patch through a conic
    // boundary; the closed-form ray-clip patch reconstruction (below) is only
    // validated for the line case, so emit just the single witness + the dims
    // there. (The polytope-polytope / polytope-plane patch, runtime_poly/contact_manifold
    // .hpp, is exact.) A full SOC patch manifold is future work.
    if (d_p == 0 || d_p >= 2) {
        out.witness_points.push_back(x.template head<3>());
        return out;
    }

    // d_p == 1: a shared line (e.g. a Cylinder edge on a hull face). Its two
    // endpoints are exact and x*-position-independent (contact_manifold.hpp).
    Eigen::JacobiSVD<Eigen::Matrix<double, NX, NX>> svdA(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, NX, 1> d1 = svdA.matrixV().col(NX - 1);
    const auto [tmin, tmax] = clipRayPrim<NX, NSOC>(s, d1, G, n_ort);
    out.witness_points.push_back(Eigen::Matrix<double, NX, 1>(x + tmin * d1).template head<3>());
    out.witness_points.push_back(Eigen::Matrix<double, NX, 1>(x + tmax * d1).template head<3>());
    return out;
}

} // namespace dcolpp::socp::runtime_poly
