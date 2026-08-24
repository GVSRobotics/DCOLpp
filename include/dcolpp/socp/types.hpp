#pragma once
// Scalar-type-generic fixed-size vector/matrix aliases and the per-primitive
// problem-matrix bundle, used by primitives.hpp/problem_matrices.hpp. T
// defaults to double; the solver internals (solver.hpp, nt_scaling.hpp,
// cone_utils.hpp) only ever run in double and keep their own double-only
// `Vec`/`Mat` aliases.

#include <Eigen/Dense>

namespace dcolpp::socp {

template <int N, typename T = double>
using TVec = Eigen::Matrix<T, N, 1>;

template <int R, int C, typename T = double>
using TMat = Eigen::Matrix<T, R, C>;

// The four problem-matrix blocks a single primitive contributes:
//   G_ort x <= h_ort   (n_ort rows, elementwise)
//   G_soc x <=_K h_soc (n_soc rows, one second-order cone)
// over a V-dimensional decision vector (V = 4 + any shape-specific extra
// decision variables, e.g. Capsule/Cylinder's axial parameter or Polygon's
// local 2D coordinate).
template <int N_ORT, int N_SOC, int V, typename T>
struct ProblemMats {
    TMat<N_ORT, V, T> G_ort;
    TVec<N_ORT, T> h_ort;
    TMat<N_SOC, V, T> G_soc;
    TVec<N_SOC, T> h_soc;
};

} // namespace dcolpp::socp
