#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/solvers/coneqp/static_solver2.jl (Kevin Tracy, MIT License).
// See NOTICE.md at the repository root for full attribution.
//
// A primal-dual interior-point solver for second-order-cone programs
//     minimize    c'x
//     subject to  Gx + s = h,  s in K
// where K = R^{n_ort}_+ x Q^{n_soc1} x Q^{n_soc2} (nonnegative orthant times
// up to two second-order cones), using Nesterov-Todd scaling and a
// Mehrotra-style predictor-corrector step. All sizes (nx, n_ort, n_soc1,
// n_soc2) are compile-time template parameters, exactly mirroring how the
// Julia source carries them via StaticArrays type parameters.

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

#include "dcolpp/socp/cone_utils.hpp"
#include "dcolpp/socp/nt_scaling.hpp"
#include "dcolpp/socp/small_llt.hpp"

namespace dcolpp::socp {

template <int nx>
using DecisionVec = Vec<nx>;

template <int n_ort, int n_soc1, int n_soc2, int nx>
using ConstraintMat = Mat<n_ort + n_soc1 + n_soc2, nx>;

template <int n_ort, int n_soc1, int n_soc2>
double ortLinesearch(const Vec<n_ort>& x, const Vec<n_ort>& dx) {
    double alpha = 1.0;
    if constexpr (n_ort > 0) {
        for (int i = 0; i < n_ort; ++i) {
            if (dx(i) < 0.0) alpha = std::min(alpha, -x(i) / dx(i));
        }
    }
    return alpha;
}

template <int n_soc>
double socLinesearch(const Vec<n_soc>& y, const Vec<n_soc>& delta) {
    static_assert(n_soc >= 1, "socLinesearch requires a nonempty SOC block");
    const auto yv = y.template tail<n_soc - 1>();
    const auto dv = delta.template tail<n_soc - 1>();

    const double nu = y(0) * y(0) - yv.squaredNorm();
    const double zeta = y(0) * delta(0) - yv.dot(dv);
    const double sqrt_nu = std::sqrt(nu);

    Vec<n_soc> rho;
    rho(0) = zeta / nu;
    rho.template tail<n_soc - 1>() =
        dv / sqrt_nu - (((zeta / sqrt_nu) + delta(0)) / (y(0) / sqrt_nu + 1.0)) * (yv / nu);

    const double rho_tail_norm = rho.template tail<n_soc - 1>().norm();
    if (rho_tail_norm > rho(0)) {
        return std::min(1.0, 1.0 / (rho_tail_norm - rho(0)));
    }
    return 1.0;
}

template <int n_ort, int n_soc1, int n_soc2>
double lineSearch(const StackVec<n_ort, n_soc1, n_soc2>& x, const StackVec<n_ort, n_soc1, n_soc2>& dx) {
    double alpha = 1.0;
    if constexpr (n_ort > 0) {
        alpha = std::min(alpha, ortLinesearch<n_ort, n_soc1, n_soc2>(x.template head<n_ort>(), dx.template head<n_ort>()));
    }
    if constexpr (n_soc1 > 0) {
        alpha = std::min(alpha, socLinesearch<n_soc1>(x.template segment<n_soc1>(n_ort), dx.template segment<n_soc1>(n_ort)));
    }
    if constexpr (n_soc2 > 0) {
        alpha = std::min(alpha, socLinesearch<n_soc2>(x.template segment<n_soc2>(n_ort + n_soc1), dx.template segment<n_soc2>(n_ort + n_soc1)));
    }
    return alpha;
}

template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> bring2cone(const StackVec<n_ort, n_soc1, n_soc2>& r) {
    double alpha = -1.0;

    if constexpr (n_ort > 0) {
        const Vec<n_ort> r_ort = r.template head<n_ort>();
        if ((r_ort.array() <= 0.0).any()) alpha = -r_ort.minCoeff();
    }
    if constexpr (n_soc1 > 0) {
        const Vec<n_soc1> r_soc1 = r.template segment<n_soc1>(n_ort);
        const double res = r_soc1(0) - r_soc1.template tail<n_soc1 - 1>().norm();
        if (res <= 0.0) alpha = std::max(alpha, -res);
    }
    if constexpr (n_soc2 > 0) {
        const Vec<n_soc2> r_soc2 = r.template segment<n_soc2>(n_ort + n_soc1);
        const double res = r_soc2(0) - r_soc2.template tail<n_soc2 - 1>().norm();
        if (res <= 0.0) alpha = std::max(alpha, -res);
    }

    if (alpha < 0.0) return r;
    return r + (1.0 + alpha) * gen_e<n_ort, n_soc1, n_soc2, double>();
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SocpInit {
    DecisionVec<nx> x;
    StackVec<n_ort, n_soc1, n_soc2> s;
    StackVec<n_ort, n_soc1, n_soc2> z;
};

template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> initializeSocp(const DecisionVec<nx>& c,
                                                    const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                    const StackVec<n_ort, n_soc1, n_soc2>& h) {
    const Mat<nx, n_ort + n_soc1 + n_soc2> GT = G.transpose();
    SmallLLT<nx> F;
    F.compute(Mat<nx, nx>(GT * G));

    const DecisionVec<nx> x_for_s = F.solve(DecisionVec<nx>(GT * h));
    const StackVec<n_ort, n_soc1, n_soc2> s_tilde = G * x_for_s - h;
    const StackVec<n_ort, n_soc1, n_soc2> s0 = bring2cone<n_ort, n_soc1, n_soc2>(s_tilde);

    const DecisionVec<nx> x0 = F.solve(DecisionVec<nx>(-c));
    const StackVec<n_ort, n_soc1, n_soc2> z_tilde = G * x0;
    const StackVec<n_ort, n_soc1, n_soc2> z0 = bring2cone<n_ort, n_soc1, n_soc2>(z_tilde);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x_for_s, s0, z0};
}

template <int n_ort, int n_soc1, int n_soc2, int nx>
struct SocpResult {
    DecisionVec<nx> x;
    StackVec<n_ort, n_soc1, n_soc2> s;
    StackVec<n_ort, n_soc1, n_soc2> z;
    int iters = 0;
    bool converged = false;
};

struct SocpOptions {
    double pdip_tol = 1e-6;
    int max_iters = 50;
};

template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpResult<n_ort, n_soc1, n_soc2, nx> solveSocp(const DecisionVec<nx>& c,
                                                 const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                 const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                 const SocpOptions& opt = SocpOptions{}) {
    auto init = initializeSocp<n_ort, n_soc1, n_soc2, nx>(c, G, h);
    DecisionVec<nx> x = init.x;
    StackVec<n_ort, n_soc1, n_soc2> s = init.s;
    StackVec<n_ort, n_soc1, n_soc2> z = init.z;

    const StackVec<n_ort, n_soc1, n_soc2> e = gen_e<n_ort, n_soc1, n_soc2, double>();

    int cone_degree = n_ort;
    if constexpr (n_soc1 > 0) cone_degree += 1;
    if constexpr (n_soc2 > 0) cone_degree += 1;

    SocpResult<n_ort, n_soc1, n_soc2, nx> result;
    const Mat<nx, n_ort + n_soc1 + n_soc2> GT = G.transpose();

    for (int main_iter = 1; main_iter <= opt.max_iters; ++main_iter) {
        const double mu = s.dot(z) / static_cast<double>(cone_degree);

        if (mu < opt.pdip_tol) {
            result.x = x;
            result.s = s;
            result.z = z;
            result.iters = main_iter;
            result.converged = true;
            return result;
        }

        NTScaling<n_ort, n_soc1, n_soc2> W = calcNTScalings<n_ort, n_soc1, n_soc2>(s, z);

        const StackVec<n_ort, n_soc1, n_soc2> lambda = W.apply(z);
        const StackVec<n_ort, n_soc1, n_soc2> lambda_lambda = cone_product<n_ort, n_soc1, n_soc2, double>(lambda, lambda);

        const DecisionVec<nx> rx = GT * z + c;
        const StackVec<n_ort, n_soc1, n_soc2> rz = s + G * x - h;

        // affine step
        const DecisionVec<nx> bx = -rx;
        StackVec<n_ort, n_soc1, n_soc2> lambda_ds =
            inverse_cone_product<n_ort, n_soc1, n_soc2, double>(lambda, -lambda_lambda);
        StackVec<n_ort, n_soc1, n_soc2> bz_tilde = W.solve(-rz - W.apply(lambda_ds));
        ConstraintMat<n_ort, n_soc1, n_soc2, nx> Gt = W.template solveMat<nx>(G);
        const Mat<nx, n_ort + n_soc1 + n_soc2> GtT = Gt.transpose();
        SmallLLT<nx> F;
        F.compute(Mat<nx, nx>(GtT * Gt));

        DecisionVec<nx> dxa = F.solve(DecisionVec<nx>(bx + GtT * bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dza = W.solve(StackVec<n_ort, n_soc1, n_soc2>(Gt * dxa - bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dsa = W.apply(lambda_ds - W.apply(dza));

        const double alpha_a = std::min(lineSearch<n_ort, n_soc1, n_soc2>(s, dsa),
                                         lineSearch<n_ort, n_soc1, n_soc2>(z, dza));
        const double rho = (s + alpha_a * dsa).dot(z + alpha_a * dza) / s.dot(z);
        const double rho_clamped = std::max(0.0, std::min(1.0, rho));
        const double sigma = rho_clamped * rho_clamped * rho_clamped;

        // centering + correcting step
        StackVec<n_ort, n_soc1, n_soc2> ds =
            -lambda_lambda - cone_product<n_ort, n_soc1, n_soc2, double>(W.solve(dsa), W.apply(dza)) + sigma * mu * e;
        lambda_ds = inverse_cone_product<n_ort, n_soc1, n_soc2, double>(lambda, ds);
        bz_tilde = W.solve(-rz - W.apply(lambda_ds));

        DecisionVec<nx> dx = F.solve(DecisionVec<nx>(bx + GtT * bz_tilde));
        StackVec<n_ort, n_soc1, n_soc2> dz = W.solve(Gt * dx - bz_tilde);
        StackVec<n_ort, n_soc1, n_soc2> ds_final = W.apply(lambda_ds - W.apply(dz));

        const double alpha = std::min(1.0, 0.99 * std::min(lineSearch<n_ort, n_soc1, n_soc2>(s, ds_final),
                                                             lineSearch<n_ort, n_soc1, n_soc2>(z, dz)));

        x += alpha * dx;
        s += alpha * ds_final;
        z += alpha * dz;

        result.iters = main_iter;
    }

    result.x = x;
    result.s = s;
    result.z = z;
    result.converged = false;
    return result;
}

} // namespace dcolpp::socp
