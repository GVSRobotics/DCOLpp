#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/solvers/coneqp/soc_utils.jl (Kevin Tracy, MIT License).
// See NOTICE.md at the repository root for full attribution.
//
// Second-order-cone (SOC) algebra used by the primal-dual interior-point
// solver: the "arrow" matrix representation of a cone element, the
// Jordan/cone product and its inverse, and the cone identity element `e`.
//
// Block sizes (n_ort ORT rows, n_soc1/n_soc2 SOC block sizes) are template
// parameters rather than runtime values, mirroring how the Julia source
// carries them as StaticArrays type parameters -- this keeps every matrix
// fixed-size and lets `if constexpr` elide code for the (common) case where
// one of the two SOC blocks is empty (e.g. a Polytope contributes no SOC
// rows at all), instead of the runtime-`if` branches the Julia code uses.

#include <Eigen/Dense>

namespace dcolpp::socp {

template <int N>
using Vec = Eigen::Matrix<double, N, 1>;
template <int R, int C>
using Mat = Eigen::Matrix<double, R, C>;

// soc_quad_J(x) = x_s^2 - dot(x_v, x_v), the SOC quadratic form. n >= 1.
template <int N>
double soc_quad_J(const Vec<N>& x) {
    static_assert(N >= 1, "soc_quad_J requires a nonempty SOC block");
    const double xs = x(0);
    const double xv2 = x.template tail<N - 1>().squaredNorm();
    return xs * xs - xv2;
}

template <int N>
Vec<N> normalize_soc(const Vec<N>& x) {
    return x / std::sqrt(soc_quad_J<N>(x));
}

// arrow(x): the "arrow" matrix Arw(x) such that Arw(x) y == soc_cone_product(x,y).
template <int N>
Mat<N, N> arrow(const Vec<N>& x) {
    static_assert(N >= 1, "arrow requires a nonempty SOC block");
    Mat<N, N> A = Mat<N, N>::Zero();
    A(0, 0) = x(0);
    A.template block<1, N - 1>(0, 1) = x.template tail<N - 1>().transpose();
    A.template block<N - 1, 1>(1, 0) = x.template tail<N - 1>();
    A.template block<N - 1, N - 1>(1, 1) = x(0) * Mat<N - 1, N - 1>::Identity();
    return A;
}

// soc_cone_product(u,v) = [u.v ; u0*v1 + v0*u1]  (the Jordan product on Q^N)
template <int N>
Vec<N> soc_cone_product(const Vec<N>& u, const Vec<N>& v) {
    static_assert(N >= 1, "soc_cone_product requires a nonempty SOC block");
    Vec<N> out;
    out(0) = u.dot(v);
    out.template tail<N - 1>() = u(0) * v.template tail<N - 1>() + v(0) * u.template tail<N - 1>();
    return out;
}

// inverse_soc_cone_product(u,w): solves soc_cone_product(u, y) = w for y.
template <int N>
Vec<N> inverse_soc_cone_product(const Vec<N>& u, const Vec<N>& w) {
    static_assert(N >= 1, "inverse_soc_cone_product requires a nonempty SOC block");
    const double u0 = u(0);
    const auto u1 = u.template tail<N - 1>();
    const double w0 = w(0);
    const auto w1 = w.template tail<N - 1>();

    const double rho = u0 * u0 - u1.squaredNorm();
    const double nu = u1.dot(w1);

    Vec<N> out;
    out(0) = (u0 * w0 - nu) / rho;
    out.template tail<N - 1>() = ((nu / u0 - w0) * u1 + (rho / u0) * w1) / rho;
    return out;
}

// The cone identity element for a block of size N (1 in the ORT slot(s) /
// the SOC "time" component, 0 elsewhere).
template <int N>
Vec<N> gen_e_block() {
    Vec<N> e = Vec<N>::Zero();
    e(0) = 1.0;
    return e;
}

// -----------------------------------------------------------------------
// Composite versions over the full [ort; soc1; soc2] stacked vector.
// -----------------------------------------------------------------------

template <int n_ort, int n_soc1, int n_soc2>
using StackVec = Vec<n_ort + n_soc1 + n_soc2>;

template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> cone_product(const StackVec<n_ort, n_soc1, n_soc2>& s,
                                              const StackVec<n_ort, n_soc1, n_soc2>& z) {
    StackVec<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) {
        out.template head<n_ort>() = s.template head<n_ort>().cwiseProduct(z.template head<n_ort>());
    }
    if constexpr (n_soc1 > 0) {
        out.template segment<n_soc1>(n_ort) =
            soc_cone_product<n_soc1>(s.template segment<n_soc1>(n_ort), z.template segment<n_soc1>(n_ort));
    }
    if constexpr (n_soc2 > 0) {
        out.template segment<n_soc2>(n_ort + n_soc1) = soc_cone_product<n_soc2>(
            s.template segment<n_soc2>(n_ort + n_soc1), z.template segment<n_soc2>(n_ort + n_soc1));
    }
    return out;
}

template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> inverse_cone_product(const StackVec<n_ort, n_soc1, n_soc2>& lambda,
                                                      const StackVec<n_ort, n_soc1, n_soc2>& v) {
    StackVec<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) {
        out.template head<n_ort>() = v.template head<n_ort>().cwiseQuotient(lambda.template head<n_ort>());
    }
    if constexpr (n_soc1 > 0) {
        out.template segment<n_soc1>(n_ort) = inverse_soc_cone_product<n_soc1>(
            lambda.template segment<n_soc1>(n_ort), v.template segment<n_soc1>(n_ort));
    }
    if constexpr (n_soc2 > 0) {
        out.template segment<n_soc2>(n_ort + n_soc1) = inverse_soc_cone_product<n_soc2>(
            lambda.template segment<n_soc2>(n_ort + n_soc1), v.template segment<n_soc2>(n_ort + n_soc1));
    }
    return out;
}

template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> gen_e() {
    StackVec<n_ort, n_soc1, n_soc2> e;
    if constexpr (n_ort > 0) e.template head<n_ort>() = Vec<n_ort>::Ones();
    if constexpr (n_soc1 > 0) e.template segment<n_soc1>(n_ort) = gen_e_block<n_soc1>();
    if constexpr (n_soc2 > 0) e.template segment<n_soc2>(n_ort + n_soc1) = gen_e_block<n_soc2>();
    return e;
}

} // namespace dcolpp::socp
