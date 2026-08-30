#pragma once
// dcolpp::socp
//
// Builds each shape's contribution (G_ort, h_ort, G_soc, h_soc) to the SOCP
//     minimize    alpha
//     subject to  G_ort [p;alpha;extras] <= h_ort
//                 G_soc [p;alpha;extras] <=_K h_soc      (one 2nd-order cone)
// whose solution is the minimum uniform scaling `alpha` that must be applied
// to both shapes (grown from a common witness point p) before they touch.
//
// Parameterized on pose g: for body 1, g = Identity; for body 2, g = g1^{-1} g2
// (relative pose from body 1 to body 2). Body 1's blocks are pose-independent
// and cached (see proximity.hpp / warm_start.hpp).

#include "dcolpp/socp/primitives.hpp"
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

// The shape's local frame placed into the pair's reference frame. With
// g = (Rg, rg) and the shape's optional mounting offset (R_offset, r_offset):
//   R = Rg * R_offset          placed rotation (the shape's axes)
//   r = rg + Rg * r_offset     placed origin
struct PlacedFrame {
    Eigen::Matrix3d R;
    Eigen::Vector3d r;
};

template <typename Shape>
PlacedFrame placeShape(const Shape& shape, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d Rg = g.block<3, 3>(0, 0);
    const Eigen::Vector3d rg = g.block<3, 1>(0, 3);
    PlacedFrame out;
    out.R = Rg * shape.R_offset;
    out.r = rg + Rg * shape.r_offset;
    return out;
}

// -------------------------------------------------------------------------
// Notation used in every block below. Decision vars: p (witness point, in
// the pair's reference frame), alpha (uniform scale), plus any
// shape-specific extras. Each shape's blocks encode "p lies inside the
// shape scaled by alpha about its local origin r" (alpha < 1 penetrating,
// == 1 touching, > 1 apart).
//   R, r  = pf.R, pf.r          placed rotation / origin
//   y     = R^T (p - r)         the witness in the shape's own local frame
//   bx    = R * (1,0,0)         placed local x-axis;  bx.(p - r) = y0
// The solver forms s = h - G x and requires s in K: ORT rows elementwise
// s >= 0; each SOC block s0 >= ||s_tail||. (Shape radii are written c.R /
// s.R below, to keep R free for the rotation.)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Capsule : n_ort=2, n_soc=4, v=5 (extra decision var: axial parameter t)
//   a ball of radius c.R swept along the local-x segment [-L/2, +L/2].
//   SOC:  || p - r - t*bx ||  <=  c.R*alpha
//   ORT:  -(L/2)*alpha  <=  t  <=  (L/2)*alpha       (t on the swept segment)
// -------------------------------------------------------------------------
inline ProblemMats<2, 4, 5> problemMatrices(const Capsule& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);

    ProblemMats<2, 4, 5> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = -c.R;
    out.G_soc.block<3, 3>(1, 0) = -Eigen::Matrix3d::Identity();
    out.G_soc.block<3, 1>(1, 4) = bx;

    out.h_soc(0) = 0.0;
    out.h_soc.segment<3>(1) = -pf.r;

    out.G_ort.setZero();
    out.G_ort(0, 3) = -c.L / 2.0; out.G_ort(0, 4) = 1;
    out.G_ort(1, 3) = -c.L / 2.0; out.G_ort(1, 4) = -1;
    out.h_ort.setZero();
    return out;
}

// -------------------------------------------------------------------------
// Cylinder : n_ort=4, n_soc=4, v=5
//   Capsule's ball-around-axis-point, plus a clamp on the witness's OWN
//   axial coordinate so the ends are flat disks, not hemispheres.
//   SOC:  || p - r - t*bx ||  <=  c.R*alpha
//   ORT:  -(L/2)*alpha  <=  t   <=  (L/2)*alpha       (axis parameter)
//         -(L/2)*alpha  <=  y0  <=  (L/2)*alpha       (flat end caps)
// -------------------------------------------------------------------------
inline ProblemMats<4, 4, 5> problemMatrices(const Cylinder& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);

    ProblemMats<4, 4, 5> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = -c.R;
    out.G_soc.block<3, 3>(1, 0) = -Eigen::Matrix3d::Identity();
    out.G_soc.block<3, 1>(1, 4) = bx;

    out.h_soc(0) = 0.0;
    out.h_soc.segment<3>(1) = -pf.r;

    out.G_ort.setZero();
    out.G_ort(0, 3) = -c.L / 2.0; out.G_ort(0, 4) = 1;
    out.G_ort(1, 3) = -c.L / 2.0; out.G_ort(1, 4) = -1;
    out.G_ort.block<1, 3>(2, 0) = (-bx).transpose(); out.G_ort(2, 3) = -c.L / 2.0;
    out.G_ort.block<1, 3>(3, 0) = bx.transpose();    out.G_ort(3, 3) = -c.L / 2.0;

    const double bxdotr = bx.dot(pf.r);
    out.h_ort(0) = 0.0;
    out.h_ort(1) = 0.0;
    out.h_ort(2) = -bxdotr;
    out.h_ort(3) = bxdotr;
    return out;
}

// -------------------------------------------------------------------------
// Cone : n_ort=1, n_soc=3, v=4
//   infinite cone (half-angle beta), cut by the flat base plane.
//   E = diag(tan(beta), 1, 1);  apex sits at y0 = -(3H/4)*alpha.
//   SOC:  sqrt(y1^2 + y2^2)  <=  tan(beta) * ( y0 + (3H/4)*alpha )
//   ORT:  y0  <=  (H/4)*alpha                          (base cap)
// -------------------------------------------------------------------------
inline ProblemMats<1, 3, 4> problemMatrices(const Cone& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const double tanb = std::tan(c.beta);
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb; E(1, 1) = 1; E(2, 2) = 1;

    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);
    const Eigen::Matrix3d ERt = E * pf.R.transpose();

    ProblemMats<1, 3, 4> out;
    out.G_soc.setZero();
    out.G_soc.block<3, 3>(0, 0) = -ERt;
    out.G_soc(0, 3) = -tanb * 3.0 * c.H / 4.0;
    out.G_soc(1, 3) = 0.0;
    out.G_soc(2, 3) = 0.0;
    out.h_soc = -ERt * pf.r;

    out.G_ort.block<1, 3>(0, 0) = bx.transpose();
    out.G_ort(0, 3) = -c.H / 4.0;
    out.h_ort(0) = bx.dot(pf.r);
    return out;
}

// -------------------------------------------------------------------------
// TruncatedCone : n_ort=2, n_soc=3, v=4
// -------------------------------------------------------------------------
// SOC block is byte-identical to Cone's (same E = diag(tanb,1,1), same
// -E*Q^T, same h_soc) -- the lateral surface is the same infinite cone. The
// only SOC difference is the alpha coefficient in row 0: apex_dist replaces
// Cone's 3H/4. The two orthant rows clip the flat caps at local x = +-L/2
// (origin at the axial midpoint).
//   tanb = (R_bottom - R_top)/L,   apex_dist = L/2 + R_top/tanb
//   SOC:  sqrt(y1^2 + y2^2)  <=  tanb * ( y0 + apex_dist*alpha )
//   ORT:  -(L/2)*alpha  <=  y0  <=  (L/2)*alpha         (both flat caps)
inline ProblemMats<2, 3, 4> problemMatrices(const TruncatedCone& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const double tanb = c.tan_beta;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb; E(1, 1) = 1; E(2, 2) = 1;

    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);
    const Eigen::Matrix3d ERt = E * pf.R.transpose();

    ProblemMats<2, 3, 4> out;
    out.G_soc.setZero();
    out.G_soc.block<3, 3>(0, 0) = -ERt;
    out.G_soc(0, 3) = -tanb * c.apex_dist;
    out.h_soc = -ERt * pf.r;

    const double half_l = c.L / 2.0;
    const double bxdotr = bx.dot(pf.r);
    out.G_ort.setZero();
    out.G_ort.block<1, 3>(0, 0) = bx.transpose();  out.G_ort(0, 3) = -half_l; // y0 <= (L/2)*alpha
    out.G_ort.block<1, 3>(1, 0) = -bx.transpose(); out.G_ort(1, 3) = -half_l; // y0 >= -(L/2)*alpha
    out.h_ort(0) = bxdotr;
    out.h_ort(1) = -bxdotr;
    return out;
}

// -------------------------------------------------------------------------
// Sphere : n_ort=0, n_soc=4, v=4
//   SOC:  || p - r ||  <=  s.R*alpha      (the alpha-scaled ball about r)
// -------------------------------------------------------------------------
inline ProblemMats<0, 4, 4> problemMatrices(const Sphere& s, const Eigen::Matrix4d& g) {
    const Eigen::Vector3d p = g.block<3, 1>(0, 3) + g.block<3, 3>(0, 0) * s.r_offset;

    ProblemMats<0, 4, 4> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = -s.R;
    out.G_soc.block<3, 3>(1, 0) = -Eigen::Matrix3d::Identity();

    out.h_soc(0) = 0.0;
    out.h_soc.segment<3>(1) = -p;
    return out;
}

// -------------------------------------------------------------------------
// Ellipsoid : n_ort=0, n_soc=4, v=4  (x' P x <= 1, U = chol(P) upper)
//   x' P x <= 1  <=>  || U x || <= 1;  scaled by alpha:
//   SOC:  || U * R^T (p - r) ||  <=  alpha
// -------------------------------------------------------------------------
inline ProblemMats<0, 4, 4> problemMatrices(const Ellipsoid& e, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(e, g);
    const Eigen::Matrix3d URt = e.U * pf.R.transpose();

    ProblemMats<0, 4, 4> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = -1;
    out.G_soc.block<3, 3>(1, 0) = -URt;

    out.h_soc(0) = 0.0;
    out.h_soc.segment<3>(1) = -(URt * pf.r);
    return out;
}

// -------------------------------------------------------------------------
// Polytope<NH> : n_ort=NH, n_soc=0, v=4  (A x <= b, locally)
//   p inside the alpha-scaled polytope { x : A x <= b } about r:
//   ORT:  A * R^T (p - r)  <=  alpha * b        (one row per half-space)
// -------------------------------------------------------------------------
template <int NH>
ProblemMats<NH, 0, 4> problemMatrices(const Polytope<NH>& poly, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, NH, 3> ARt = poly.A * pf.R.transpose();

    ProblemMats<NH, 0, 4> out;
    out.G_ort.block(0, 0, NH, 3) = ARt;
    out.G_ort.block(0, 3, NH, 1) = -poly.b;
    out.h_ort = ARt * pf.r;
    return out;
}

// -------------------------------------------------------------------------
// Polygon<NH> : n_ort=NH, n_soc=4, v=6 (extra: 2 local in-plane coords u)
//   a 2D polygon { A u <= b } in the local z=0 plane, Minkowski-summed with
//   a ball of radius poly.R (the cushion). Rtilde = R's first two columns
//   lifts u into 3D.
//   ORT:  A u  <=  alpha * b                        (in-plane half-spaces)
//   SOC:  || p - r - Rtilde*u ||  <=  poly.R*alpha  (cushion ball at the in-plane point)
// -------------------------------------------------------------------------
template <int NH>
ProblemMats<NH, 4, 6> problemMatrices(const Polygon<NH>& poly, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, 3, 2> Rtilde = pf.R.template block<3, 2>(0, 0);

    ProblemMats<NH, 4, 6> out;
    out.G_ort.setZero();
    out.G_ort.block(0, 3, NH, 1) = -poly.b;
    out.G_ort.block(0, 4, NH, 2) = poly.A;
    out.h_ort.setZero();

    out.G_soc.setZero();
    out.G_soc(0, 3) = -poly.R;
    out.G_soc.template block<3, 3>(1, 0) = -Eigen::Matrix3d::Identity();
    out.G_soc.template block<3, 2>(1, 4) = Rtilde;

    out.h_soc(0) = 0.0;
    out.h_soc.template segment<3>(1) = -pf.r;
    return out;
}

} // namespace dcolpp::socp
