#pragma once
// dcolpp::socp::runtime_poly -- combined problem for PolytopeX (shape 1) vs a curved
// primitive (shape 2). The primitive's blocks are the fixed-size
// problemMatrices(prim, g) output; only the hull contributes dynamic rows.
//
// Column layout matches combine_problem_matrices.hpp for V1 = 4 (hull has no
// extras), V2 in {4, 5}: NX = V2, shared [p; alpha] at columns 0..3, the
// primitive's own extras (Capsule / Cylinder axial t) at columns [4, V2).

#include <Eigen/Dense>

#include "dcolpp/socp/problem_matrices.hpp" // fixed problemMatrices(prim, g), ProblemMats
#include "dcolpp/socp/socp_init.hpp"        // ExtraDim, extrasGuess, boundingSphere
#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"
#include "dcolpp/socp/runtime_poly/solver_soc.hpp"

namespace dcolpp::socp::runtime_poly {

// Combined PolytopeX-vs-primitive SOCP. NX = V2, NSOC = N_SOC2.
template <int NX, int NSOC>
struct CombinedProblemPrim {
    Eigen::Matrix<double, NX, 1> c;
    ConstraintMatX<NX> G;
    ConeVecX h;
    int n_ort1 = 0; // hull rows
    int n_ort = 0;  // hull rows + primitive ORT rows
};

template <int N_ORT2, int N_SOC2, int V2>
auto combineHullPrim(const ProblemMatsX& hull, const dcolpp::socp::ProblemMats<N_ORT2, N_SOC2, V2>& prim) {
    constexpr int NX = V2;
    const int nh = static_cast<int>(hull.G_ort.rows());
    const int n_ort = nh + N_ORT2;
    const int ns = n_ort + N_SOC2;

    CombinedProblemPrim<NX, N_SOC2> out;
    out.c.setZero();
    out.c(3) = 1.0;
    out.n_ort1 = nh;
    out.n_ort = n_ort;

    out.G.resize(ns, NX);
    out.G.setZero();
    out.G.topRows(nh).template leftCols<4>() = hull.G_ort;
    out.G.middleRows(nh, N_ORT2) = prim.G_ort;
    out.G.template bottomRows<N_SOC2>() = prim.G_soc;

    out.h.resize(ns);
    out.h.head(nh) = hull.h_ort;
    out.h.segment(nh, N_ORT2) = prim.h_ort;
    out.h.template tail<N_SOC2>() = prim.h_soc;
    return out;
}

// geometricPrimalGuess (socp_init.hpp), shape1 = hull (non-halfspace, no
// extras), shape2 = primitive (e2 = ExtraDim).
template <typename Prim>
Eigen::Matrix<double, 4 + dcolpp::socp::ExtraDim<Prim>::value, 1> geometricPrimalGuessPrim(
    const BoundingSphere& b1, const Prim& prim, const Eigen::Matrix4d& g) {
    constexpr int e2 = dcolpp::socp::ExtraDim<Prim>::value;
    constexpr int NX = 4 + e2;
    const BoundingSphere b2 = dcolpp::socp::boundingSphere(prim);
    const double eps = 1e-9;

    const Eigen::Vector3d r1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d r2 = g.block<3, 1>(0, 3);
    const Eigen::Vector3d rvec = r2 - r1;
    const double dist = rvec.norm();
    const Eigen::Vector3d rhat = (dist > 1e-9) ? Eigen::Vector3d(rvec / dist) : Eigen::Vector3d(1, 0, 0);

    const double alpha_min = dist / std::max(b1.outer + b2.outer, eps);
    double alpha_max = dist / std::max(b1.inner + b2.inner, eps);
    if (alpha_max < alpha_min) alpha_max = 2.0 * alpha_min + eps;
    const double alpha0 = std::sqrt(std::max(alpha_min, eps) * std::max(alpha_max, eps));
    const Eigen::Vector3d p0 = r1 + (alpha0 * b1.outer) * rhat;

    Eigen::Matrix<double, NX, 1> x0;
    x0.template head<3>() = p0;
    x0(3) = alpha0;
    if constexpr (e2 > 0) x0.template segment<e2>(4) = dcolpp::socp::extrasGuess(prim, g, p0);
    return x0;
}

// Cold solve (Geometric strategy) for a hull-primitive pair.
template <int NX, int NSOC, typename Prim>
SocpResultSoc<NX, NSOC> solveProximitySocpPrim(const BoundingSphere& b1, const Prim& prim, const Eigen::Matrix4d& g,
                                               const CombinedProblemPrim<NX, NSOC>& cp, const SocpOptions& opt) {
    if (opt.init_strategy == SocpInitStrategy::Geometric) {
        const Eigen::Matrix<double, NX, 1> x0 = geometricPrimalGuessPrim(b1, prim, g);
        const SocpInitSoc<NX, NSOC> init = initializeSocpFromGuessSoc<NX, NSOC>(cp.c, cp.G, cp.h, x0, cp.n_ort);
        return solveSocpSoc<NX, NSOC>(cp.c, cp.G, cp.h, cp.n_ort, opt, &init);
    }
    return solveSocpSoc<NX, NSOC>(cp.c, cp.G, cp.h, cp.n_ort, opt, nullptr);
}

} // namespace dcolpp::socp::runtime_poly
