#pragma once
// dcolpp::socp second-order-cone utilities.
//
// Second-order-cone (SOC) algebra used by the primal-dual interior-point
// solver: the "arrow" matrix representation of a cone element, the
// Jordan/cone product and its inverse, and the cone identity element `e`.
//
// Block sizes (n_ort ORT rows, n_soc1/n_soc2 SOC block sizes) are template
// parameters rather than runtime values. This keeps every matrix fixed-size
// and lets `if constexpr` elide code for the common case where one of the two
// SOC blocks is empty, e.g. a Polytope contributes no SOC rows at all.

#include <Eigen/Dense>
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

// Every function below is a plain scalar loop over the block's own N. These
// hot paths avoid Eigen expression overhead; N is small (3 or 4 for every
// shape in this library), so the loops fully unroll under optimization.

// soc_quad_J(x) = x_s^2 - dot(x_v, x_v), the SOC quadratic form. n >= 1.
// The template is the mechanism for "compile-time-known but varies by instantiation." 
template <int N>
DCOLPP_INLINE double soc_quad_J(const Vec<N>& x) {
    static_assert(N >= 1, "soc_quad_J requires a nonempty SOC block");
    double xv2 = 0.0;
    for (int i = 1; i < N; ++i) xv2 += x(i) * x(i);
    return x(0) * x(0) - xv2;
}

template <int N>
DCOLPP_INLINE Vec<N> normalize_soc(const Vec<N>& x) {
    const double inv_norm = 1.0 / std::sqrt(soc_quad_J<N>(x));
    Vec<N> out;
    for (int i = 0; i < N; ++i) out(i) = x(i) * inv_norm;
    return out;
}

// arrow(x): the "arrow" matrix Arw(x) such that Arw(x) y == soc_cone_product(x,y) (Jordan product).
template <int N>
DCOLPP_INLINE Mat<N, N> arrow(const Vec<N>& x) {
    static_assert(N >= 1, "arrow requires a nonempty SOC block");
    Mat<N, N> A = Mat<N, N>::Zero();
    A(0, 0) = x(0);
    for (int i = 1; i < N; ++i) {
        A(0, i) = x(i);
        A(i, 0) = x(i);
        A(i, i) = x(0);
    }
    return A;
}

// soc_cone_product(u,v) = [u.v ; u0*v1 + v0*u1]  (the Jordan product on Q^N)
template <int N>
DCOLPP_INLINE Vec<N> soc_cone_product(const Vec<N>& u, const Vec<N>& v) {
    static_assert(N >= 1, "soc_cone_product requires a nonempty SOC block");
    double dot = 0.0;
    for (int i = 0; i < N; ++i) dot += u(i) * v(i);
    Vec<N> out;
    out(0) = dot;
    const double u0 = u(0), v0 = v(0);
    for (int i = 1; i < N; ++i) out(i) = u0 * v(i) + v0 * u(i);
    return out;
}

// inverse_soc_cone_product(u,w): solves soc_cone_product(u, y) = w for y.
template <int N>
DCOLPP_INLINE Vec<N> inverse_soc_cone_product(const Vec<N>& u, const Vec<N>& w) {
    static_assert(N >= 1, "inverse_soc_cone_product requires a nonempty SOC block");
    const double u0 = u(0);
    const double w0 = w(0);
    double u1_sq = 0.0, nu = 0.0;
    for (int i = 1; i < N; ++i) {
        u1_sq += u(i) * u(i);
        nu += u(i) * w(i);
    }
    const double rho = u0 * u0 - u1_sq;
    const double inv_rho = 1.0 / rho;

    Vec<N> out;
    out(0) = (u0 * w0 - nu) * inv_rho;
    const double c1 = (nu / u0 - w0);
    const double c2 = rho / u0;
    for (int i = 1; i < N; ++i) out(i) = (c1 * u(i) + c2 * w(i)) * inv_rho;
    return out;
}

// The cone identity element for a block of size N (1 in the ORT slot(s) /
// the SOC "time" component, 0 elsewhere).
template <int N>
DCOLPP_INLINE Vec<N> gen_e_block() {
    return Vec<N>::Unit(0); // (1, 0, ..., 0) -- setup-only, not a hot path
}

// -----------------------------------------------------------------------
// Composite versions over the full [ort; soc1; soc2] stacked vector.
// -----------------------------------------------------------------------

template <int n_ort, int n_soc1, int n_soc2>
using StackVec = Vec<n_ort + n_soc1 + n_soc2>;

template <int n_ort, int n_soc1, int n_soc2>
DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> cone_product(const StackVec<n_ort, n_soc1, n_soc2>& s,
                                              const StackVec<n_ort, n_soc1, n_soc2>& z) {
    StackVec<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) {
        for (int i = 0; i < n_ort; ++i) out(i) = s(i) * z(i);
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
DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> inverse_cone_product(const StackVec<n_ort, n_soc1, n_soc2>& lambda,
                                                      const StackVec<n_ort, n_soc1, n_soc2>& v) {
    StackVec<n_ort, n_soc1, n_soc2> out;
    if constexpr (n_ort > 0) {
        for (int i = 0; i < n_ort; ++i) out(i) = v(i) / lambda(i);
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
DCOLPP_INLINE StackVec<n_ort, n_soc1, n_soc2> gen_e() {
    StackVec<n_ort, n_soc1, n_soc2> e;
    if constexpr (n_ort > 0) e.template head<n_ort>() = Vec<n_ort>::Ones();
    if constexpr (n_soc1 > 0) e.template segment<n_soc1>(n_ort) = gen_e_block<n_soc1>();
    if constexpr (n_soc2 > 0) e.template segment<n_soc2>(n_ort + n_soc1) = gen_e_block<n_soc2>();
    return e;
}

} // namespace dcolpp::socp
