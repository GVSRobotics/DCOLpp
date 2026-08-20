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
//
// Scalar type T defaults to `double` (every call site in the solver itself
// uses that); it exists at all so `dcolpp::socp::kkt_R` (proximity.hpp,
// Phase 3) can reuse the exact same cone_product formula with T = Dual6,
// matching how soc_utils.jl is already generic over T in the original.

#include <Eigen/Dense>
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

template <int N>
using Vec = TVec<N, double>;
template <int R, int C>
using Mat = TMat<R, C, double>;

// soc_quad_J(x) = x_s^2 - dot(x_v, x_v), the SOC quadratic form. n >= 1.
template <int N, typename T = double>
T soc_quad_J(const TVec<N, T>& x) {
    static_assert(N >= 1, "soc_quad_J requires a nonempty SOC block");
    const T xs = x(0);
    const T xv2 = x.template tail<N - 1>().squaredNorm();
    return xs * xs - xv2;
}

template <int N, typename T = double>
TVec<N, T> normalize_soc(const TVec<N, T>& x) {
    using std::sqrt;
    return x / sqrt(soc_quad_J<N, T>(x));
}

// arrow(x): the "arrow" matrix Arw(x) such that Arw(x) y == soc_cone_product(x,y).
template <int N, typename T = double>
TMat<N, N, T> arrow(const TVec<N, T>& x) {
    static_assert(N >= 1, "arrow requires a nonempty SOC block");
    TMat<N, N, T> A = TMat<N, N, T>::Zero();
    A(0, 0) = x(0);
    A.template block<1, N - 1>(0, 1) = x.template tail<N - 1>().transpose();
    A.template block<N - 1, 1>(1, 0) = x.template tail<N - 1>();
    A.template block<N - 1, N - 1>(1, 1) = x(0) * TMat<N - 1, N - 1, T>::Identity();
    return A;
}

// soc_cone_product(u,v) = [u.v ; u0*v1 + v0*u1]  (the Jordan product on Q^N)
template <int N, typename T = double>
TVec<N, T> soc_cone_product(const TVec<N, T>& u, const TVec<N, T>& v) {
    static_assert(N >= 1, "soc_cone_product requires a nonempty SOC block");
    TVec<N, T> out;
    out(0) = u.dot(v);
    out.template tail<N - 1>() = u(0) * v.template tail<N - 1>() + v(0) * u.template tail<N - 1>();
    return out;
}

// inverse_soc_cone_product(u,w): solves soc_cone_product(u, y) = w for y.
template <int N, typename T = double>
TVec<N, T> inverse_soc_cone_product(const TVec<N, T>& u, const TVec<N, T>& w) {
    static_assert(N >= 1, "inverse_soc_cone_product requires a nonempty SOC block");
    const T u0 = u(0);
    const auto u1 = u.template tail<N - 1>();
    const T w0 = w(0);
    const auto w1 = w.template tail<N - 1>();

    const T rho = u0 * u0 - u1.squaredNorm();
    const T nu = u1.dot(w1);

    TVec<N, T> out;
    out(0) = (u0 * w0 - nu) / rho;
    out.template tail<N - 1>() = ((nu / u0 - w0) * u1 + (rho / u0) * w1) / rho;
    return out;
}

// The cone identity element for a block of size N (1 in the ORT slot(s) /
// the SOC "time" component, 0 elsewhere).
template <int N, typename T = double>
TVec<N, T> gen_e_block() {
    TVec<N, T> e = TVec<N, T>::Zero();
    e(0) = T(1);
    return e;
}

// -----------------------------------------------------------------------
// Composite versions over the full [ort; soc1; soc2] stacked vector.
// -----------------------------------------------------------------------

template <int n_ort, int n_soc1, int n_soc2, typename T = double>
using StackVecT = TVec<n_ort + n_soc1 + n_soc2, T>;

template <int n_ort, int n_soc1, int n_soc2>
using StackVec = StackVecT<n_ort, n_soc1, n_soc2, double>;

template <int n_ort, int n_soc1, int n_soc2, typename T = double>
StackVecT<n_ort, n_soc1, n_soc2, T> cone_product(const StackVecT<n_ort, n_soc1, n_soc2, T>& s,
                                                  const StackVecT<n_ort, n_soc1, n_soc2, T>& z) {
    StackVecT<n_ort, n_soc1, n_soc2, T> out;
    if constexpr (n_ort > 0) {
        out.template head<n_ort>() = s.template head<n_ort>().cwiseProduct(z.template head<n_ort>());
    }
    if constexpr (n_soc1 > 0) {
        out.template segment<n_soc1>(n_ort) =
            soc_cone_product<n_soc1, T>(s.template segment<n_soc1>(n_ort), z.template segment<n_soc1>(n_ort));
    }
    if constexpr (n_soc2 > 0) {
        out.template segment<n_soc2>(n_ort + n_soc1) = soc_cone_product<n_soc2, T>(
            s.template segment<n_soc2>(n_ort + n_soc1), z.template segment<n_soc2>(n_ort + n_soc1));
    }
    return out;
}

template <int n_ort, int n_soc1, int n_soc2, typename T = double>
StackVecT<n_ort, n_soc1, n_soc2, T> inverse_cone_product(const StackVecT<n_ort, n_soc1, n_soc2, T>& lambda,
                                                          const StackVecT<n_ort, n_soc1, n_soc2, T>& v) {
    StackVecT<n_ort, n_soc1, n_soc2, T> out;
    if constexpr (n_ort > 0) {
        out.template head<n_ort>() = v.template head<n_ort>().cwiseQuotient(lambda.template head<n_ort>());
    }
    if constexpr (n_soc1 > 0) {
        out.template segment<n_soc1>(n_ort) = inverse_soc_cone_product<n_soc1, T>(
            lambda.template segment<n_soc1>(n_ort), v.template segment<n_soc1>(n_ort));
    }
    if constexpr (n_soc2 > 0) {
        out.template segment<n_soc2>(n_ort + n_soc1) = inverse_soc_cone_product<n_soc2, T>(
            lambda.template segment<n_soc2>(n_ort + n_soc1), v.template segment<n_soc2>(n_ort + n_soc1));
    }
    return out;
}

template <int n_ort, int n_soc1, int n_soc2, typename T = double>
StackVecT<n_ort, n_soc1, n_soc2, T> gen_e() {
    StackVecT<n_ort, n_soc1, n_soc2, T> e;
    if constexpr (n_ort > 0) e.template head<n_ort>() = TVec<n_ort, T>::Ones();
    if constexpr (n_soc1 > 0) e.template segment<n_soc1>(n_ort) = gen_e_block<n_soc1, T>();
    if constexpr (n_soc2 > 0) e.template segment<n_soc2>(n_ort + n_soc1) = gen_e_block<n_soc2, T>();
    return e;
}

} // namespace dcolpp::socp
