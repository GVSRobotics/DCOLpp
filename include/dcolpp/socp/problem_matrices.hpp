#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/problem_matrices.jl (Kevin Tracy, MIT License). See NOTICE.md.
//
// Builds each shape's contribution (G_ort, h_ort, G_soc, h_soc) to the SOCP
//     minimize    alpha
//     subject to  G_ort [p;alpha;extras] <= h_ort
//                 G_soc [p;alpha;extras] <=_K h_soc      (one 2nd-order cone)
// whose solution is the minimum uniform scaling `alpha` that must be applied
// to both shapes (grown from a common witness point p) before they touch.
//
// Reparameterized on a single `Eigen::Matrix4d g`: the pose of THIS shape's
// local frame relative to the pair's reference frame (frame-1). The
// original Julia source instead took a world-frame (r, quaternion) or
// (r, MRP) pair per shape and duplicated every function below for both
// attitude parameterizations; here there is one function per shape, and
// body 1 in a pair is simply called with g = Identity (see proximity.hpp),
// matching iDCOL's `ProblemData::g = g1^{-1} g2` convention.

#include "dcolpp/socp/primitives.hpp"
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

// The shape's local origin/axes placed into the pair's reference frame:
// Q = R * Q_offset, r = p + R * r_offset (R,p = g's rotation/translation).
struct PlacedFrame {
    Eigen::Matrix3d Q;
    Eigen::Vector3d r;
};

template <typename Shape>
PlacedFrame placeShape(const Shape& shape, const Eigen::Matrix4d& g) {
    const Eigen::Matrix3d R = g.block<3, 3>(0, 0);
    const Eigen::Vector3d p = g.block<3, 1>(0, 3);
    PlacedFrame out;
    out.Q = R * shape.Q_offset;
    out.r = p + R * shape.r_offset;
    return out;
}

// -------------------------------------------------------------------------
// Capsule : n_ort=2, n_soc=4, v=5 (extra decision var: axial parameter t)
// -------------------------------------------------------------------------
inline ProblemMats<2, 4, 5> problemMatrices(const Capsule& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);

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
// -------------------------------------------------------------------------
inline ProblemMats<4, 4, 5> problemMatrices(const Cylinder& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);

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
// -------------------------------------------------------------------------
inline ProblemMats<1, 3, 4> problemMatrices(const Cone& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const double tanb = std::tan(c.beta);
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb; E(1, 1) = 1; E(2, 2) = 1;

    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);
    const Eigen::Matrix3d EQt = E * pf.Q.transpose();

    ProblemMats<1, 3, 4> out;
    out.G_soc.setZero();
    out.G_soc.block<3, 3>(0, 0) = -EQt;
    out.G_soc(0, 3) = -tanb * 3.0 * c.H / 4.0;
    out.G_soc(1, 3) = 0.0;
    out.G_soc(2, 3) = 0.0;
    out.h_soc = -EQt * pf.r;

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
// only SOC difference is the alpha coefficient in row 0: the radius bound is
// tanb*(y0 + apex_dist*alpha), with apex_dist the origin-to-virtual-apex
// distance (Cone's is 3H/4). The two orthant rows clip the flat caps at
// local x = +-L/2 (origin at the axial midpoint), exactly Cylinder's
// rows 2/3 +-bx pattern.
inline ProblemMats<2, 3, 4> problemMatrices(const TruncatedCone& c, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(c, g);
    const double tanb = c.tan_beta;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb; E(1, 1) = 1; E(2, 2) = 1;

    const Eigen::Vector3d bx = pf.Q * Eigen::Vector3d(1, 0, 0);
    const Eigen::Matrix3d EQt = E * pf.Q.transpose();

    ProblemMats<2, 3, 4> out;
    out.G_soc.setZero();
    out.G_soc.block<3, 3>(0, 0) = -EQt;
    out.G_soc(0, 3) = -tanb * c.apex_dist;
    out.h_soc = -EQt * pf.r;

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
// -------------------------------------------------------------------------
inline ProblemMats<0, 4, 4> problemMatrices(const Ellipsoid& e, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(e, g);
    const Eigen::Matrix3d UQt = e.U * pf.Q.transpose();

    ProblemMats<0, 4, 4> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = -1;
    out.G_soc.block<3, 3>(1, 0) = -UQt;

    out.h_soc(0) = 0.0;
    out.h_soc.segment<3>(1) = -(UQt * pf.r);
    return out;
}

// -------------------------------------------------------------------------
// Polytope<NH> : n_ort=NH, n_soc=0, v=4  (A x <= b, locally)
// -------------------------------------------------------------------------
template <int NH>
ProblemMats<NH, 0, 4> problemMatrices(const Polytope<NH>& poly, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, NH, 3> AQt = poly.A * pf.Q.transpose();

    ProblemMats<NH, 0, 4> out;
    out.G_ort.block(0, 0, NH, 3) = AQt;
    out.G_ort.block(0, 3, NH, 1) = -poly.b;
    out.h_ort = AQt * pf.r;
    return out;
}

// -------------------------------------------------------------------------
// Polygon<NH> : n_ort=NH, n_soc=4, v=6 (extra: 2 local in-plane coordinates)
// -------------------------------------------------------------------------
template <int NH>
ProblemMats<NH, 4, 6> problemMatrices(const Polygon<NH>& poly, const Eigen::Matrix4d& g) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, 3, 2> Qtilde = pf.Q.template block<3, 2>(0, 0);

    ProblemMats<NH, 4, 6> out;
    out.G_ort.setZero();
    out.G_ort.block(0, 3, NH, 1) = -poly.b;
    out.G_ort.block(0, 4, NH, 2) = poly.A;
    out.h_ort.setZero();

    out.G_soc.setZero();
    out.G_soc(0, 3) = -poly.R;
    out.G_soc.template block<3, 3>(1, 0) = -Eigen::Matrix3d::Identity();
    out.G_soc.template block<3, 2>(1, 4) = Qtilde;

    out.h_soc(0) = 0.0;
    out.h_soc.template segment<3>(1) = -pf.r;
    return out;
}

} // namespace dcolpp::socp
