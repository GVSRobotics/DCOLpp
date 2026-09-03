#pragma once
// dcolpp::socp::runtime_poly -- PolytopeX: a polytope whose half-space count is fixed
// only at construction (e.g. one piece of a mesh's convex decomposition).
// Purely additive: the fixed-size primitives.hpp / Polytope<NH> are untouched.
//
// No padding -- A holds exactly the real faces, so the contact KKT system
// keeps full rank and the analytic derivatives stay well-posed. Half-space
// storage is inline and capped at kMaxHalfspaces (runtime_poly/config.hpp): no heap on
// the query hot path.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/config.hpp"
#include "dcolpp/socp/primitives.hpp" // dcolpp::socp::BoundingSphere

namespace dcolpp::socp::runtime_poly {

using dcolpp::socp::BoundingSphere;

namespace detail {

// Runtime-NH counterparts of dcolpp::socp::detail::polytopeVertices /
// polytopeBoundingSphere. Byte-for-byte the same math; loop bounds come from
// A.rows() instead of the template NH. Run once, in the PolytopeX ctor.
template <typename DA, typename Db>
std::vector<Eigen::Vector3d> polytopeVertices(const Eigen::MatrixBase<DA>& A, const Eigen::MatrixBase<Db>& b) {
    const int nh = static_cast<int>(A.rows());
    std::vector<Eigen::Vector3d> vs;
    const double bscale = 1.0 + b.cwiseAbs().maxCoeff();
    const double feas_tol = 1e-9 * bscale;
    const double dup_tol2 = (1e-7 * bscale) * (1e-7 * bscale);
    for (int i = 0; i < nh; ++i) {
        for (int j = i + 1; j < nh; ++j) {
            for (int k = j + 1; k < nh; ++k) {
                const Eigen::Vector3d ai = A.row(i).transpose();
                const Eigen::Vector3d aj = A.row(j).transpose();
                const Eigen::Vector3d ak = A.row(k).transpose();
                const Eigen::Vector3d aj_x_ak = aj.cross(ak);
                const double det = ai.dot(aj_x_ak);
                const double scale = ai.norm() * aj.norm() * ak.norm();
                if (scale == 0.0 || std::abs(det) < 1e-9 * scale) continue; // planes not independent
                const Eigen::Vector3d y =
                    (b(i) * aj_x_ak + b(j) * ak.cross(ai) + b(k) * ai.cross(aj)) / det;
                if (!((A * y).array() <= b.array() + feas_tol).all()) continue; // outside another face
                bool seen = false;
                for (const auto& v : vs)
                    if ((v - y).squaredNorm() <= dup_tol2) { seen = true; break; }
                if (!seen) vs.push_back(y);
            }
        }
    }
    return vs;
}

template <typename DA, typename Db>
BoundingSphere polytopeBoundingSphere(const Eigen::MatrixBase<DA>& A, const Eigen::MatrixBase<Db>& b,
                                      const std::vector<Eigen::Vector3d>& verts) {
    const int nh = static_cast<int>(A.rows());
    double inner = std::numeric_limits<double>::infinity();
    double maxface = 0.0;
    for (int i = 0; i < nh; ++i) {
        const double d = b(i) / A.row(i).norm();
        inner = std::min(inner, d);
        maxface = std::max(maxface, d);
    }
    double outer = 0.0;
    for (const auto& v : verts) outer = std::max(outer, v.norm());
    if (!(outer > 0.0)) outer = std::sqrt(3.0) * maxface;

    return {inner, outer};
}

} // namespace detail

// { x : A x <= b }, A.rows() runtime (4 .. kMaxHalfspaces). Never a halfspace,
// never strictly convex -- the runtime_poly contact path treats it as such directly.
struct PolytopeX {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int MaxNH = kMaxHalfspaces;
    using AMat = Eigen::Matrix<double, Eigen::Dynamic, 3, 0, MaxNH, 3>;
    using BVec = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxNH, 1>;

    AMat A;
    BVec b;
    std::vector<Eigen::Vector3d> vertices; // unique feasible vertices, local frame -- precomputed once
    const BoundingSphere bounding_sphere;

    template <typename DA, typename Db>
    PolytopeX(const Eigen::MatrixBase<DA>& A_, const Eigen::MatrixBase<Db>& b_)
        : A(A_),
          b(b_),
          vertices(detail::polytopeVertices(A_, b_)),
          bounding_sphere(detail::polytopeBoundingSphere(A_, b_, vertices)) {
        eigen_assert(A_.cols() == 3 && A_.rows() == b_.rows() && A_.rows() >= 4 && A_.rows() <= MaxNH);
    }

    int nh() const { return static_cast<int>(A.rows()); }
};

} // namespace dcolpp::socp::runtime_poly
