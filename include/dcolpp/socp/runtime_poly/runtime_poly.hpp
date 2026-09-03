#pragma once
// dcolpp::socp::runtime_poly -- runtime-face-count polytope (PolytopeX) collision path.
// Umbrella include: pull this for the whole runtime_poly API.
//
// WHY THIS EXISTS
// --------------
// The fixed-size pipeline templates every shape's arity into its type:
// Polytope<NH> / Polygon<NH> bake the face / edge count into the SOCP
// dimensions at compile time. That is ideal when the geometry is known when
// the engine is built, but a consuming engine that imports rigid bodies as
// *convex decompositions of arbitrary meshes* (serial arms, quadrupeds, any
// URDF/MJCF mesh collision) gets pieces whose face count is known only at
// model-assembly time. Padding a large Polytope<Nmax> with slack half-spaces
// makes the contact KKT system rank-deficient and breaks the analytic
// derivatives, so the polytope needs a genuine runtime face count.
//
// PolytopeX is that: a convex polytope { x : A x <= b } whose half-space count
// is fixed only at construction, stored inline and capped at kMaxHalfspaces
// (runtime_poly/config.hpp) -- no heap on the query hot path, no padding, KKT stays
// full rank. Approach: a parallel solver templated only on the small
// compile-time dims (nx in {4,5}, one SOC block in {0,3,4}); only the ORT row
// count is runtime. All SOC / Cholesky / per-primitive derivative helpers are
// the fixed-size versions, reused verbatim. The fixed dcolpp::socp pipeline
// (Polytope<NH>, Polygon<NH>, every primitive) is byte-untouched.
//
// SUPPORTED (dcolpp::socp::runtime_poly::...):
//   proximity / alphaGradient / proximityJacobian
//   proximityContact / proximityContactJacobian
// for these ordered pairs (shape1, shape2):
//   (PolytopeX, PolytopeX)
//   (Plane,     PolytopeX)                       -- Plane is always shape 1
//   (PolytopeX, {Sphere|Ellipsoid|Capsule|Cylinder|Cone|TruncatedCone})
// plus opt-in contact manifold / degeneracy (SocpOptions::
// compute_contact_manifold / compute_degeneracy_info), and a warm-start handle
// (ContactWarmStateX) for the pure-ORT pairs (PolytopeX-PolytopeX,
// Plane-PolytopeX). Numerically matched to the fixed Polytope<NH> path (see
// tests/test_polytope_dynamic.cpp): values ~1e-14, Jacobians ~1e-8, manifold
// points bit-identical. Speed ~1.2x the fixed path at N>=48, ~1.9x at N=6.
//
// NOT DONE YET
// -----------
//  * PolygonX -- a runtime-edge-count counterpart of the fixed Polygon<NH>
//    (a 2D convex polygon in a local plane, Minkowski-summed with a cushion
//    ball of radius R; problemMatrices(Polygon<NH>) -> ProblemMats<NH,4,6>,
//    v=6 with two extra in-plane decision vars). No consuming-engine need has
//    come up -- decomposed mesh pieces are 3D polytopes, not cushioned
//    polygons -- so it is deliberately skipped. Adding it would mirror
//    PolytopeX: a PolygonX shape (runtime_poly/primitives.hpp), its dynamic
//    ProblemMats + combine (nx=6, one SOC block), the fixed
//    polygonXiDerivative / polygonHessianFrozen reused, and the nx=6 SOC
//    solver path (solver_soc.hpp already generalises on <NX,NSOC>).
//  * SOC-patch contact manifold for d_p >= 2 on a PolytopeX-vs-curved-
//    primitive pair (a flat primitive feature flush on a hull face): returns
//    the single witness + correct degeneracy dims for now. The fixed
//    contact_manifold.hpp also caps its patch reconstruction at d_p == 2.
//  * warm-start for PolytopeX-vs-curved-primitive (the SOC warmStartInit*
//    variants are unported); those queries run cold.

#include "dcolpp/socp/runtime_poly/config.hpp"
#include "dcolpp/socp/runtime_poly/contact.hpp"
#include "dcolpp/socp/runtime_poly/contact_manifold.hpp"
#include "dcolpp/socp/runtime_poly/contact_manifold_prim.hpp"
#include "dcolpp/socp/runtime_poly/contact_prim.hpp"
#include "dcolpp/socp/runtime_poly/primitives.hpp"
#include "dcolpp/socp/runtime_poly/proximity.hpp"
#include "dcolpp/socp/runtime_poly/proximity_prim.hpp"
#include "dcolpp/socp/runtime_poly/warm_start.hpp"
