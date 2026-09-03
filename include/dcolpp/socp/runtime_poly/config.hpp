#pragma once
// dcolpp::socp::runtime_poly -- compile-time configuration for the runtime-face-count
// (PolytopeX) collision path. Nothing here affects the fixed-size pipeline.

namespace dcolpp::socp::runtime_poly {

// Upper bound on a PolytopeX's half-space count. Storage is inline
// Matrix<double, Dynamic, C, 0, DCOLPP_MAX_HALFSPACES, C> -- no heap on the
// query hot path, capped here. 128 covers any practical convex-decomposition
// piece (V-HACD / CoACD; MuJoCo's maxhullvert default is 64). Override with
// -DDCOLPP_MAX_HALFSPACES.
#ifndef DCOLPP_MAX_HALFSPACES
#define DCOLPP_MAX_HALFSPACES 128
#endif

inline constexpr int kMaxHalfspaces = DCOLPP_MAX_HALFSPACES;

// Largest combined stacked-cone row count the runtime_poly solver's inline buffers must
// hold: two hulls (2*kMaxHalfspaces), or one hull plus a primitive partner's
// worst-case ORT + SOC rows (Cylinder: 4 ORT + 4 SOC). +8 covers both.
inline constexpr int kMaxOrtRows = 2 * kMaxHalfspaces + 8;

} // namespace dcolpp::socp::runtime_poly
