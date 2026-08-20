#pragma once
// Scalar-type-generic fixed-size vector/matrix aliases used by the parts of
// dcolpp::socp that must work over both `double` (forward solve) and
// `Dual6` (differentiation): primitives' geometry, problem-matrix
// construction, and the KKT residual. The solver internals themselves
// (dcolpp/socp/solver.hpp, nt_scaling.hpp, cone_utils.hpp) only ever run in
// double and keep their own double-only `Vec`/`Mat` aliases.

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

// Elementwise scalar-type cast of a whole ProblemMats bundle. Used to lift
// the reference body's (always-double, always-identity-pose) problem
// matrices into Dual6 with an exactly-zero derivative, instead of pointlessly
// re-deriving them with Dual6 arithmetic -- body 1 in a pair never moves, so
// its true derivative w.r.t. the pair's relative-pose twist is zero anyway.
template <typename T2, int N_ORT, int N_SOC, int V, typename T1>
ProblemMats<N_ORT, N_SOC, V, T2> castProblemMats(const ProblemMats<N_ORT, N_SOC, V, T1>& in) {
    ProblemMats<N_ORT, N_SOC, V, T2> out;
    out.G_ort = in.G_ort.template cast<T2>();
    out.h_ort = in.h_ort.template cast<T2>();
    out.G_soc = in.G_soc.template cast<T2>();
    out.h_soc = in.h_soc.template cast<T2>();
    return out;
}

} // namespace dcolpp::socp
