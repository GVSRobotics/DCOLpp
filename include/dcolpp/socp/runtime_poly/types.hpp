#pragma once
// dcolpp::socp::runtime_poly -- bounded-dynamic vector / matrix aliases for the
// PolytopeX solver path. Row count is runtime; storage is inline (capped at
// kMaxOrtRows / N), so nothing here allocates on the query hot path.

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/config.hpp"

namespace dcolpp::socp::runtime_poly {

// Stacked-cone vector s / z / h: runtime length, inline storage up to
// kMaxOrtRows. (M1/M2 are ORT-only, so length == n_ort.)
using StackVecX = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, kMaxOrtRows, 1>;

// Constraint matrix G: runtime rows, nx columns fixed at compile time.
template <int NX>
using ConstraintMatX = Eigen::Matrix<double, Eigen::Dynamic, NX, 0, kMaxOrtRows, NX>;

// nx-wide objects (decision vector, its xi-Jacobian, ...): fully fixed-size,
// nx is 4 (hull-hull) / 5 (hull-Capsule/Cylinder) -- never dynamic.
template <int NX>
using DecisionVecX = Eigen::Matrix<double, NX, 1>;

} // namespace dcolpp::socp::runtime_poly
