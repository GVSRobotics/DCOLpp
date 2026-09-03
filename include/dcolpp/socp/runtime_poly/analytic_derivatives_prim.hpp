#pragma once
// dcolpp::socp::runtime_poly -- IFT sensitivity for PolytopeX (shape 1) vs a curved
// primitive (shape 2). The primitive's per-shape xi-derivative and frozen
// Hessian are the fixed-size functions from analytic_derivatives.hpp, reused
// verbatim; only the combine (dynamic hull row count) and the block-elimination
// solve are runtime here. Column layout: V1 = 4, NX = V2, NSOC = N_SOC2.

#include <Eigen/Dense>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/analytic_derivatives.hpp" // shapeXiDerivative, shapeHessianFrozen, arrow
#include "dcolpp/socp/runtime_poly/problem_matrices_prim.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::arrow;
using dcolpp::socp::shapeHessianFrozen;
using dcolpp::socp::shapeXiDerivative;
using dcolpp::socp::ShapeXiDerivative;

template <int NX, int NSOC>
struct SensitivityResultPrim {
    Eigen::Matrix<double, NX, 6> dx;
    MatX6<kMaxOrtRows> ds; // ns x 6
    MatX6<kMaxOrtRows> dz;
};

// combineXiJacobian (analytic_derivatives.hpp) for N_SOC1 = 0, V1 = 4,
// N_ORT1 = nh runtime.
template <int N_ORT2, int N_SOC2, int V2>
struct CombinedXiJacPrim {
    Eigen::Matrix<double, V2, 6> dR1_dxi = Eigen::Matrix<double, V2, 6>::Zero();
    MatX6<kMaxOrtRows> dR2_dxi; // ns x 6
    MatX6<kMaxOrtRows> q;       // ns x 6
    int ns = 0;
};

template <int N_ORT2, int N_SOC2, int V2>
CombinedXiJacPrim<N_ORT2, N_SOC2, V2> combineXiJacPrim(const ShapeXiDerivative<N_ORT2, N_SOC2, V2>& s2, int nh,
                                                      const StackVecX& z) {
    const int n_ort = nh + N_ORT2;
    const int ns = n_ort + N_SOC2;
    const int soc2_row = n_ort;

    CombinedXiJacPrim<N_ORT2, N_SOC2, V2> out;
    out.ns = ns;
    out.q.setZero(ns, 6);
    if constexpr (N_ORT2 > 0) out.q.middleRows(nh, N_ORT2) = s2.dHort - s2.dGortX;
    if constexpr (N_SOC2 > 0) out.q.template bottomRows<N_SOC2>() = s2.dHsoc - s2.dGsocX;

    const Eigen::Matrix<double, V2, 6> s2_r1 = s2.dGortTz + s2.dGsocTz;
    out.dR1_dxi.template topRows<4>() = s2_r1.template topRows<4>();
    if constexpr (V2 > 4) out.dR1_dxi.template block<V2 - 4, 6>(4, 0) = s2_r1.template bottomRows<V2 - 4>();

    // dR2/dxi = Arrow(z) q, block-diagonal.
    out.dR2_dxi.setZero(ns, 6);
    out.dR2_dxi.topRows(n_ort) = z.head(n_ort).asDiagonal() * out.q.topRows(n_ort);
    if constexpr (N_SOC2 > 0)
        out.dR2_dxi.template bottomRows<N_SOC2>() =
            arrow<N_SOC2>(Vec<N_SOC2>(z.template tail<N_SOC2>())) * out.q.template bottomRows<N_SOC2>();
    return out;
}

// computeSocpSensitivityFromXiJac (analytic_derivatives.hpp), N_SOC1 = 0.
template <int N_ORT2, int N_SOC2, int V2>
SensitivityResultPrim<V2, N_SOC2> sensitivityFromXiJacPrim(const CombinedXiJacPrim<N_ORT2, N_SOC2, V2>& xi,
                                                          const StackVecX& s, const StackVecX& z,
                                                          const ConstraintMatX<V2>& G, int nh) {
    constexpr int NX = V2;
    const int n_ort = nh + N_ORT2;
    const int ns = xi.ns;

    const Eigen::Matrix<double, NX, 6> r1 = -xi.dR1_dxi;
    const MatX6<kMaxOrtRows> r2 = -xi.dR2_dxi;

    // S \ Z and S \ r2. ORT: elementwise. SOC: arrow(s) factor solve.
    MatX6<kMaxOrtRows> Sr2(ns, 6);
    Sr2.topRows(n_ort) = r2.topRows(n_ort).array().colwise() / s.head(n_ort).array();

    Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX> SZG(ns, NX);
    SZG.topRows(n_ort) = (z.head(n_ort).cwiseQuotient(s.head(n_ort))).asDiagonal() * G.topRows(n_ort);

    if constexpr (N_SOC2 > 0) {
        const Eigen::Matrix<double, N_SOC2, N_SOC2> Ssoc = arrow<N_SOC2>(Vec<N_SOC2>(s.template tail<N_SOC2>()));
        const Eigen::Matrix<double, N_SOC2, N_SOC2> Zsoc = arrow<N_SOC2>(Vec<N_SOC2>(z.template tail<N_SOC2>()));
        SmallLLT<N_SOC2> Sf;
        Sf.compute(Ssoc);
        const Eigen::Matrix<double, N_SOC2, N_SOC2> SZsoc = Sf.solve(Zsoc);
        SZG.template bottomRows<N_SOC2>() = SZsoc * G.template bottomRows<N_SOC2>();
        Sr2.template bottomRows<N_SOC2>() = Sf.solve(r2.template bottomRows<N_SOC2>());
    }

    const Eigen::Matrix<double, NX, NX> A = G.transpose() * SZG;
    const Eigen::Matrix<double, NX, 6> rhs = r1 - G.transpose() * Sr2;

    SensitivityResultPrim<NX, N_SOC2> out;
    out.dx = A.partialPivLu().solve(rhs);
    out.dz.noalias() = SZG * out.dx + Sr2;
    out.ds.noalias() = xi.q - G * out.dx;
    return out;
}

// --- shape-2 slices + public results -----------------------------------

template <typename Prim, int N_ORT2, int N_SOC2, int V2>
ShapeXiDerivative<N_ORT2, N_SOC2, V2> primXiDeriv(const Prim& prim, const Eigen::Matrix4d& g,
                                                  const Eigen::Matrix<double, V2, 1>& x2, const StackVecX& z,
                                                  int nh) {
    const Vec<N_ORT2> z_ort2 = z.segment(nh, N_ORT2);
    const Vec<N_SOC2> z_soc2 = z.template tail<N_SOC2>();
    return shapeXiDerivative(prim, g, x2, z_ort2, z_soc2);
}

template <int V2, int NSOC, typename Prim>
Eigen::Matrix<double, 4, 6> diffSocpPrim(const Prim& prim, const Eigen::Matrix4d& g,
                                         const Eigen::Matrix<double, V2, 1>& x, const StackVecX& s,
                                         const StackVecX& z, const ConstraintMatX<V2>& G, int nh) {
    constexpr int N_ORT2 = decltype(dcolpp::socp::problemMatrices(prim, g).G_ort)::RowsAtCompileTime;
    const auto s2 = primXiDeriv<Prim, N_ORT2, NSOC, V2>(prim, g, x, z, nh);
    const auto xi = combineXiJacPrim<N_ORT2, NSOC, V2>(s2, nh, z);
    const auto sens = sensitivityFromXiJacPrim<N_ORT2, NSOC, V2>(xi, s, z, G, nh);
    return sens.dx.template topRows<4>();
}

template <int V2, int NSOC, typename Prim>
Eigen::Matrix<double, 1, 6> proximityGradientPrim(const Prim& prim, const Eigen::Matrix4d& g,
                                                  const Eigen::Matrix<double, V2, 1>& x, const StackVecX& z, int nh) {
    constexpr int N_ORT2 = decltype(dcolpp::socp::problemMatrices(prim, g).G_ort)::RowsAtCompileTime;
    const auto s2 = primXiDeriv<Prim, N_ORT2, NSOC, V2>(prim, g, x, z, nh);
    const auto xi = combineXiJacPrim<N_ORT2, NSOC, V2>(s2, nh, z);
    return -(xi.q.transpose() * z.head(xi.ns));
}

struct ContactJacBundlePrim {
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 3, 6> normal_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
};

template <int V2, int NSOC, typename Prim>
ContactJacBundlePrim contactJacobianBundlePrim(const Prim& prim, const Eigen::Matrix4d& g,
                                               const Eigen::Matrix<double, V2, 1>& x, const StackVecX& s,
                                               const StackVecX& z, const ConstraintMatX<V2>& G, int nh) {
    constexpr int N_ORT2 = decltype(dcolpp::socp::problemMatrices(prim, g).G_ort)::RowsAtCompileTime;
    const Vec<N_ORT2> z_ort2 = z.segment(nh, N_ORT2);
    const Vec<NSOC> z_soc2 = z.template tail<NSOC>();
    const auto s2 = shapeXiDerivative(prim, g, x, z_ort2, z_soc2);
    const auto xi = combineXiJacPrim<N_ORT2, NSOC, V2>(s2, nh, z);
    const auto sens = sensitivityFromXiJacPrim<N_ORT2, NSOC, V2>(xi, s, z, G, nh);

    ContactJacBundlePrim out;
    out.jacobian = sens.dx.template topRows<4>();
    out.grad = -(xi.q.transpose() * z.head(xi.ns));

    const Eigen::Matrix<double, V2, 6> r1 = -xi.dR1_dxi;
    const Eigen::Matrix<double, 6, 6> H_frozen = shapeHessianFrozen(prim, g, x, z_ort2, z_soc2);
    const Eigen::Matrix<double, 6, 6> H =
        H_frozen - r1.transpose() * sens.dx - xi.q.transpose() * sens.dz;

    const Eigen::Vector3d grad_v = out.grad.template tail<3>().transpose();
    const Eigen::Matrix<double, 3, 6> dgradv_dxi = H.template bottomRows<3>();
    const Eigen::Matrix3d Rg = g.block<3, 3>(0, 0);
    const Eigen::Vector3d u = Rg * grad_v;
    const double unorm = u.norm();
    const Eigen::Matrix<double, 3, 6> du_dxi = se3::dRotatedVectorDXi(g, grad_v) + Rg * dgradv_dxi;
    const Eigen::Vector3d nrm = u / unorm;
    const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - nrm * nrm.transpose();
    out.normal_jacobian = (proj / unorm) * du_dxi;
    return out;
}

} // namespace dcolpp::socp::runtime_poly
