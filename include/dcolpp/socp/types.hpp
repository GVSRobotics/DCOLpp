#pragma once
// Fixed-size, double-only vector/matrix aliases and the per-primitive
// problem-matrix bundle. Everything in dcolpp::socp is double -- no
// autodiff scalar type anywhere in this codebase (DEVIATIONS.md SS4a) --
// so these are plain aliases, not scalar-type-generic.

#include <Eigen/Dense>

// Forces inlining of the small hot-path SOC/cone functions (cone_utils.hpp,
// solver.hpp, nt_scaling.hpp, small_llt.hpp) that solveSocp's main loop
// calls many times per iteration. Measured to matter: at -O3, GCC was
// leaving these as real out-of-line `call`s (confirmed by inspecting the
// generated assembly -- e.g. NTScaling::apply/solve, lineSearch,
// cone_product each still had live `call` instructions inside solveSocp's
// body) despite them being tiny and always_inline-eligible, unlike the
// Julia port's source, where every one of these is marked `@inline` and
// Julia's compiler reliably honors it.
#if defined(__GNUC__) || defined(__clang__)
#define DCOLPP_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define DCOLPP_INLINE __forceinline
#else
#define DCOLPP_INLINE inline
#endif

namespace dcolpp::socp {

template <int N>
using Vec = Eigen::Matrix<double, N, 1>;

template <int R, int C>
using Mat = Eigen::Matrix<double, R, C>;

// The four problem-matrix blocks a single primitive contributes:
//   G_ort x <= h_ort   (n_ort rows, elementwise)
//   G_soc x <=_K h_soc (n_soc rows, one second-order cone)
// over a V-dimensional decision vector (V = 4 + any shape-specific extra
// decision variables, e.g. Capsule/Cylinder's axial parameter or Polygon's
// local 2D coordinate).
template <int N_ORT, int N_SOC, int V>
struct ProblemMats {
    Mat<N_ORT, V> G_ort;
    Vec<N_ORT> h_ort;
    Mat<N_SOC, V> G_soc;
    Vec<N_SOC> h_soc;
};

} // namespace dcolpp::socp
