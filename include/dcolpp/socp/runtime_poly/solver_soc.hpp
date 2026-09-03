#pragma once
// dcolpp::socp::runtime_poly -- interior-point SOCP solver for a PolytopeX vs a curved
// primitive (Sphere / Ellipsoid / Capsule / Cylinder / Cone / TruncatedCone):
// a dynamic ORT block (the hull's half-spaces plus the primitive's own ORT
// rows) followed by ONE fixed second-order cone block of size NSOC in {3,4},
// over an NX-wide decision vector (NX in {4,5}).
//
// Only the ORT row count is runtime. NSOC / NX are compile-time, so every SOC
// primitive (socNTScaling / arrow / soc_cone_product / socLinesearch) and the
// NX x NX Newton factor (SmallLLT<NX>) are the fixed-size versions reused
// verbatim from solver.hpp / cone_utils.hpp / nt_scaling.hpp. Operation order
// matches solver.hpp 1:1 so a PolytopeX-primitive pair reproduces the fixed
// Polytope<NH>-primitive pair.

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/runtime_poly/types.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/small_llt.hpp"
#include "dcolpp/socp/solver.hpp" // SocpOptions, SocpInitStrategy, socLinesearch, gramLower

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::arrow;
using dcolpp::socp::inverse_soc_cone_product;
using dcolpp::socp::soc_cone_product;
using dcolpp::socp::socLinesearch;
using dcolpp::socp::socNTScaling;

// A stacked [ort(dynamic); soc(NSOC)] vector, inline storage.
using ConeVecX = StackVecX; // length n_ort + NSOC at runtime

// gen_e for [ort; soc]: ones on ORT, (1,0,..) on the SOC block.
template <int NSOC>
inline ConeVecX genE(int n_ort) {
    ConeVecX e = ConeVecX::Ones(n_ort + NSOC);
    if constexpr (NSOC > 0) e.tail(NSOC - 1).setZero();
    return e;
}

template <int NSOC>
inline ConeVecX coneProduct(const ConeVecX& s, const ConeVecX& z, int n_ort) {
    ConeVecX out(n_ort + NSOC);
    out.head(n_ort) = s.head(n_ort).cwiseProduct(z.head(n_ort));
    if constexpr (NSOC > 0)
        out.template tail<NSOC>() =
            soc_cone_product<NSOC>(Vec<NSOC>(s.template tail<NSOC>()), Vec<NSOC>(z.template tail<NSOC>()));
    return out;
}

template <int NSOC>
inline ConeVecX inverseConeProduct(const ConeVecX& lambda, const ConeVecX& v, int n_ort) {
    ConeVecX out(n_ort + NSOC);
    out.head(n_ort) = v.head(n_ort).cwiseQuotient(lambda.head(n_ort));
    if constexpr (NSOC > 0)
        out.template tail<NSOC>() = inverse_soc_cone_product<NSOC>(Vec<NSOC>(lambda.template tail<NSOC>()),
                                                                  Vec<NSOC>(v.template tail<NSOC>()));
    return out;
}

// bring2cone (solver.hpp): lift the worst per-block margin to 1.
template <int NSOC>
inline ConeVecX bring2coneSoc(const ConeVecX& r, int n_ort) {
    double m = r.head(n_ort).minCoeff();
    if constexpr (NSOC > 0) {
        const auto b = r.template tail<NSOC>();
        m = std::min(m, b(0) - b.template tail<NSOC - 1>().norm());
    }
    if (m > 0.0) return r;
    return r + (1.0 - m) * genE<NSOC>(n_ort);
}

// pushToRelativeMargin (socp_init.hpp): ORT absolute floor 0.05; SOC relative.
template <int NSOC>
inline ConeVecX pushToRelativeMarginSoc(const ConeVecX& r, int n_ort, double margin_frac) {
    double lambda = std::max(0.0, 0.05 - r.head(n_ort).minCoeff());
    if constexpr (NSOC > 0) {
        const auto b = r.template tail<NSOC>();
        const double tail_norm = b.template tail<NSOC - 1>().norm();
        lambda = std::max(lambda, tail_norm / (1.0 - margin_frac) - b(0));
    }
    if (lambda <= 0.0) return r;
    return r + lambda * genE<NSOC>(n_ort);
}

// The NT scaling operator W: diagonal on ORT, socNTScaling on the SOC block.
template <int NSOC>
struct NTScalingSoc {
    StackVecX ort;                   // sqrt(s./z) elementwise, length n_ort
    Eigen::Matrix<double, NSOC, NSOC> soc = Eigen::Matrix<double, NSOC, NSOC>::Zero();
    SmallLLT<NSOC> soc_fact;
    int n_ort = 0;

    ConeVecX apply(const ConeVecX& g) const {
        ConeVecX out(n_ort + NSOC);
        out.head(n_ort) = g.head(n_ort).cwiseProduct(ort);
        if constexpr (NSOC > 0) out.template tail<NSOC>() = soc * g.template tail<NSOC>();
        return out;
    }
    ConeVecX solve(const ConeVecX& g) const {
        ConeVecX out(n_ort + NSOC);
        out.head(n_ort) = g.head(n_ort).cwiseQuotient(ort);
        if constexpr (NSOC > 0) out.template tail<NSOC>() = soc_fact.solve(Vec<NSOC>(g.template tail<NSOC>()));
        return out;
    }
    template <int NX>
    Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX> solveMat(
        const Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX>& G) const {
        Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX> out(n_ort + NSOC, NX);
        for (int i = 0; i < n_ort; ++i) out.row(i) = G.row(i) / ort(i);
        if constexpr (NSOC > 0) out.template bottomRows<NSOC>() = soc_fact.solve(G.template bottomRows<NSOC>());
        return out;
    }
};

template <int NSOC>
inline NTScalingSoc<NSOC> calcNTScalingsSoc(const ConeVecX& s, const ConeVecX& z, int n_ort) {
    NTScalingSoc<NSOC> W;
    W.n_ort = n_ort;
    W.ort = (s.head(n_ort).cwiseQuotient(z.head(n_ort))).cwiseSqrt();
    if constexpr (NSOC > 0) {
        W.soc = socNTScaling<NSOC>(Vec<NSOC>(s.template tail<NSOC>()), Vec<NSOC>(z.template tail<NSOC>()));
        W.soc_fact.compute(W.soc);
    }
    return W;
}

// line search over [ort; soc].
template <int NSOC>
inline double lineSearchSoc(const ConeVecX& x, const ConeVecX& dx, int n_ort) {
    double a = ortLinesearch(StackVecX(x.head(n_ort)), StackVecX(dx.head(n_ort)));
    if constexpr (NSOC > 0)
        a = std::min(a, socLinesearch<NSOC>(Vec<NSOC>(x.template tail<NSOC>()), Vec<NSOC>(dx.template tail<NSOC>())));
    return a;
}

template <int NX, int NSOC>
struct SocpInitSoc {
    Eigen::Matrix<double, NX, 1> x;
    ConeVecX s;
    ConeVecX z;
};

template <int NX, int NSOC>
struct SocpResultSoc {
    Eigen::Matrix<double, NX, 1> x = Eigen::Matrix<double, NX, 1>::Zero();
    ConeVecX s;
    ConeVecX z;
    int iters = 0;
    bool converged = false;
    double mu = 0.0;
};

// initializeSocpFromGuess (socp_init.hpp) for the [ort; soc] layout.
template <int NX, int NSOC>
inline SocpInitSoc<NX, NSOC> initializeSocpFromGuessSoc(const Eigen::Matrix<double, NX, 1>& c,
                                                       const ConstraintMatX<NX>& G, const ConeVecX& h,
                                                       const Eigen::Matrix<double, NX, 1>& x0, int n_ort) {
    constexpr double kMarginFrac = 0.05;
    const int ns = n_ort + NSOC;

    const ConeVecX s_tilde = h - G * x0;
    const ConeVecX s0 = pushToRelativeMarginSoc<NSOC>(s_tilde, n_ort, kMarginFrac);

    ConeVecX z_pref = ConeVecX::Zero(ns);
    if constexpr (NSOC > 0) {
        Vec<NSOC> blk = s_tilde.template tail<NSOC>();
        blk.template tail<NSOC - 1>() *= -1.0;
        z_pref.template tail<NSOC>() = blk;
    }

    SmallLLT<NX> F;
    F.compute(Eigen::Matrix<double, NX, NX>(G.transpose() * G));
    const Eigen::Matrix<double, NX, 1> w =
        F.solve(Eigen::Matrix<double, NX, 1>(-c - G.transpose() * z_pref));
    const ConeVecX z_exact = z_pref + G * w;
    const ConeVecX z0 = pushToRelativeMarginSoc<NSOC>(z_exact, n_ort, kMarginFrac);

    return {x0, s0, z0};
}

// solveSocp (solver.hpp), [ort(dynamic); soc(NSOC)] layout.
template <int NX, int NSOC>
inline SocpResultSoc<NX, NSOC> solveSocpSoc(const Eigen::Matrix<double, NX, 1>& c, const ConstraintMatX<NX>& G,
                                            const ConeVecX& h, int n_ort, const SocpOptions& opt,
                                            const SocpInitSoc<NX, NSOC>* init_hint) {
    const int ns = n_ort + NSOC;
    Eigen::Matrix<double, NX, 1> x = init_hint ? init_hint->x : Eigen::Matrix<double, NX, 1>::Zero();
    ConeVecX s = init_hint ? init_hint->s : ConeVecX::Ones(ns);
    ConeVecX z = init_hint ? init_hint->z : ConeVecX::Ones(ns);

    const ConeVecX e = genE<NSOC>(n_ort);
    int cone_degree = n_ort;
    if constexpr (NSOC > 0) cone_degree += 1;
    const auto GT = G.transpose();

    SocpResultSoc<NX, NSOC> result;
    result.s = s;
    result.z = z;

    for (int main_iter = 1; main_iter <= opt.max_iters; ++main_iter) {
        const double mu = s.dot(z) / static_cast<double>(cone_degree);
        const Eigen::Matrix<double, NX, 1> rx = GT * z + c;
        const ConeVecX rz = s + G * x - h;

        const bool mu_ok = mu < opt.pdip_tol;
        const bool first_iter_residuals_ok =
            (main_iter > 1) || (rx.norm() < opt.pdip_tol && rz.norm() < opt.pdip_tol);
        if (mu_ok && first_iter_residuals_ok) {
            result.x = x;
            result.s = s;
            result.z = z;
            result.iters = main_iter;
            result.converged = true;
            result.mu = mu;
            return result;
        }

        const NTScalingSoc<NSOC> W = calcNTScalingsSoc<NSOC>(s, z, n_ort);
        const ConeVecX lambda = W.apply(z);
        const ConeVecX lambda_lambda = coneProduct<NSOC>(lambda, lambda, n_ort);

        // affine
        const Eigen::Matrix<double, NX, 1> bx = -rx;
        ConeVecX lambda_ds = inverseConeProduct<NSOC>(lambda, ConeVecX(-lambda_lambda), n_ort);
        ConeVecX bz_tilde = W.solve(ConeVecX(-rz - W.apply(lambda_ds)));
        const auto Gt = W.template solveMat<NX>(G);
        const auto GtT = Gt.transpose();
        SmallLLT<NX> F;
        F.compute(Eigen::Matrix<double, NX, NX>(Gt.transpose() * Gt));

        Eigen::Matrix<double, NX, 1> dxa = F.solve(Eigen::Matrix<double, NX, 1>(bx + GtT * bz_tilde));
        ConeVecX dza = W.solve(ConeVecX(Gt * dxa - bz_tilde));
        ConeVecX dsa = W.apply(ConeVecX(lambda_ds - W.apply(dza)));

        const double alpha_a =
            std::min(lineSearchSoc<NSOC>(s, dsa, n_ort), lineSearchSoc<NSOC>(z, dza, n_ort));
        const double rho = (s + alpha_a * dsa).dot(z + alpha_a * dza) / s.dot(z);
        const double rho_clamped = std::max(0.0, std::min(1.0, rho));
        const double sigma = rho_clamped * rho_clamped * rho_clamped;

        // centering + correcting
        ConeVecX ds = -lambda_lambda - coneProduct<NSOC>(W.solve(dsa), W.apply(dza), n_ort) + sigma * mu * e;
        lambda_ds = inverseConeProduct<NSOC>(lambda, ds, n_ort);
        bz_tilde = W.solve(ConeVecX(-rz - W.apply(lambda_ds)));

        Eigen::Matrix<double, NX, 1> dx = F.solve(Eigen::Matrix<double, NX, 1>(bx + GtT * bz_tilde));
        ConeVecX dz = W.solve(ConeVecX(Gt * dx - bz_tilde));
        ConeVecX ds_final = W.apply(ConeVecX(lambda_ds - W.apply(dz)));

        const double alpha = std::min(
            1.0, 0.99 * std::min(lineSearchSoc<NSOC>(s, ds_final, n_ort), lineSearchSoc<NSOC>(z, dz, n_ort)));

        x += alpha * dx;
        s += alpha * ds_final;
        z += alpha * dz;
        result.iters = main_iter;
    }

    result.x = x;
    result.s = s;
    result.z = z;
    result.converged = false;
    result.mu = s.dot(z) / static_cast<double>(cone_degree);
    return result;
}

} // namespace dcolpp::socp::runtime_poly
