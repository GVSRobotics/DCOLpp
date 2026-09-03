#pragma once
// dcolpp::socp::runtime_poly -- PolytopeX's SOCP contribution and the hull-hull
// combined problem. Mirrors problem_matrices.hpp / combine_problem_matrices.hpp
// for the case where one or both shapes have a runtime ORT row count.
//
// M1/M2: PolytopeX vs PolytopeX only (n_soc = 0, nx = 4). M3 adds the
// PolytopeX-vs-primitive layouts (dynamic ORT + one fixed SOC block).

#include <Eigen/Dense>

#include "dcolpp/socp/primitives.hpp" // dcolpp::socp::Plane
#include "dcolpp/socp/runtime_poly/primitives.hpp"
#include "dcolpp/socp/runtime_poly/types.hpp"

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::Plane;

// One PolytopeX's blocks: G_ort (nh x 4) x <= h_ort, elementwise. No SOC.
//   A R^T (p - r) <= alpha b   ->   [A R^T | -b] [p; alpha] <= A R^T r
struct ProblemMatsX {
    ConstraintMatX<4> G_ort;
    StackVecX h_ort;
};

inline ProblemMatsX problemMatrices(const PolytopeX& poly, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
    const Eigen::Vector3d r = g.block<3, 1>(0, 3);
    const int nh = poly.nh();

    Eigen::Matrix<double, Eigen::Dynamic, 3, 0, PolytopeX::MaxNH, 3> ARt(nh, 3);
    ARt.noalias() = poly.A * R.transpose();

    ProblemMatsX out;
    out.G_ort.resize(nh, 4);
    out.G_ort.leftCols<3>() = ARt;
    out.G_ort.col(3) = -poly.b;
    out.h_ort.noalias() = ARt * r;
    return out;
}

// Plane (half-space, always shape 1): normal . p <= d, does NOT scale with
// alpha. One ORT row, in the ProblemMatsX layout so combineProblemMatrices
// stacks it exactly like a hull's rows.
inline ProblemMatsX planeProblemMatrix(const Plane& plane) {
    ProblemMatsX out;
    out.G_ort.setZero(1, 4);
    out.G_ort.block<1, 3>(0, 0) = plane.normal.transpose();
    out.h_ort.resize(1);
    out.h_ort(0) = plane.d;
    return out;
}

// proximity.hpp applyPlaneFlip: if body 2's centre is on the -normal side,
// negate the plane's ORT row (row 0) so body 2 is always outside the chosen
// half-space. Returns true when it flipped.
inline bool applyPlaneFlipX(const Plane& plane, const Eigen::Matrix4d& g, ConstraintMatX<4>& G, StackVecX& h) {
    if (plane.normal.dot(g.block<3, 1>(0, 3)) - plane.d < 0.0) {
        G.row(0) = -G.row(0);
        h(0) = -h(0);
        return true;
    }
    return false;
}

// Combined hull-hull SOCP: minimize alpha s.t. G [p;alpha] <= h (elementwise),
// G = [P1.G_ort; P2.G_ort], h = [P1.h_ort; P2.h_ort]. n_ort1 kept for the
// derivative row split (analytic_derivatives.hpp).
struct CombinedProblemX {
    static constexpr int nx = 4;
    Eigen::Vector4d c;
    ConstraintMatX<4> G;
    StackVecX h;
    int n_ort1 = 0; // rows belonging to shape 1
    int n_ort = 0;  // total rows
};

inline CombinedProblemX combineProblemMatrices(const ProblemMatsX& P1, const ProblemMatsX& P2) {
    const int n1 = static_cast<int>(P1.G_ort.rows());
    const int n2 = static_cast<int>(P2.G_ort.rows());
    const int ns = n1 + n2;

    CombinedProblemX out;
    out.c = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    out.n_ort1 = n1;
    out.n_ort = ns;

    out.G.resize(ns, 4);
    out.G.topRows(n1) = P1.G_ort;
    out.G.bottomRows(n2) = P2.G_ort;

    out.h.resize(ns);
    out.h.head(n1) = P1.h_ort;
    out.h.tail(n2) = P2.h_ort;
    return out;
}

} // namespace dcolpp::socp::runtime_poly
