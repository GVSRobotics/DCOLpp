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
// Reparameterized on a single templated `Eigen::Matrix<T,4,4> g`: the pose
// of THIS shape's local frame relative to the pair's reference frame
// (frame-1). The original Julia source instead took a world-frame
// (r, quaternion) or (r, MRP) pair per shape and duplicated every function
// below for both attitude parameterizations; here there is one function per
// shape, and body 1 in a pair is simply called with g = Identity (see
// proximity.hpp), matching iDCOL's `ProblemData::g = g1^{-1} g2` convention.

#include "dcolpp/socp/primitives.hpp"
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

// The shape's local origin/axes placed into the pair's reference frame:
// Q = R * Q_offset, r = p + R * r_offset (R,p = g's rotation/translation).
template <typename T, typename Shape>
struct PlacedFrame {
    TMat<3, 3, T> Q;
    TVec<3, T> r;
};

template <typename T, typename Shape>
PlacedFrame<T, Shape> placeShape(const Shape& shape, const TMat<4, 4, T>& g) {
    const TMat<3, 3, T> R = g.template block<3, 3>(0, 0);
    const TVec<3, T> p = g.template block<3, 1>(0, 3);
    PlacedFrame<T, Shape> out;
    out.Q = R * shape.Q_offset.template cast<T>();
    out.r = p + R * shape.r_offset.template cast<T>();
    return out;
}

// -------------------------------------------------------------------------
// Capsule : n_ort=2, n_soc=4, v=5 (extra decision var: axial parameter t)
// -------------------------------------------------------------------------
template <typename T>
ProblemMats<2, 4, 5, T> problemMatrices(const Capsule& c, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T>(c, g);
    const TVec<3, T> bx = pf.Q * TVec<3, T>(T(1), T(0), T(0));

    ProblemMats<2, 4, 5, T> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = T(-c.R);
    out.G_soc.template block<3, 3>(1, 0) = -TMat<3, 3, T>::Identity();
    out.G_soc.template block<3, 1>(1, 4) = bx;

    out.h_soc(0) = T(0);
    out.h_soc.template segment<3>(1) = -pf.r;

    out.G_ort.setZero();
    out.G_ort(0, 3) = T(-c.L / 2.0); out.G_ort(0, 4) = T(1);
    out.G_ort(1, 3) = T(-c.L / 2.0); out.G_ort(1, 4) = T(-1);
    out.h_ort.setZero();
    return out;
}

// -------------------------------------------------------------------------
// Cylinder : n_ort=4, n_soc=4, v=5
// -------------------------------------------------------------------------
template <typename T>
ProblemMats<4, 4, 5, T> problemMatrices(const Cylinder& c, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T>(c, g);
    const TVec<3, T> bx = pf.Q * TVec<3, T>(T(1), T(0), T(0));

    ProblemMats<4, 4, 5, T> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = T(-c.R);
    out.G_soc.template block<3, 3>(1, 0) = -TMat<3, 3, T>::Identity();
    out.G_soc.template block<3, 1>(1, 4) = bx;

    out.h_soc(0) = T(0);
    out.h_soc.template segment<3>(1) = -pf.r;

    out.G_ort.setZero();
    out.G_ort(0, 3) = T(-c.L / 2.0); out.G_ort(0, 4) = T(1);
    out.G_ort(1, 3) = T(-c.L / 2.0); out.G_ort(1, 4) = T(-1);
    out.G_ort.template block<1, 3>(2, 0) = (-bx).transpose(); out.G_ort(2, 3) = T(-c.L / 2.0);
    out.G_ort.template block<1, 3>(3, 0) = bx.transpose();    out.G_ort(3, 3) = T(-c.L / 2.0);

    const T bxdotr = bx.dot(pf.r);
    out.h_ort(0) = T(0);
    out.h_ort(1) = T(0);
    out.h_ort(2) = -bxdotr;
    out.h_ort(3) = bxdotr;
    return out;
}

// -------------------------------------------------------------------------
// Cone : n_ort=1, n_soc=3, v=4
// -------------------------------------------------------------------------
template <typename T>
ProblemMats<1, 3, 4, T> problemMatrices(const Cone& c, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T>(c, g);
    const double tanb = std::tan(c.beta);
    TMat<3, 3, T> E = TMat<3, 3, T>::Zero();
    E(0, 0) = T(tanb); E(1, 1) = T(1); E(2, 2) = T(1);

    const TVec<3, T> bx = pf.Q * TVec<3, T>(T(1), T(0), T(0));
    const TMat<3, 3, T> EQt = E * pf.Q.transpose();

    ProblemMats<1, 3, 4, T> out;
    out.G_soc.setZero();
    out.G_soc.template block<3, 3>(0, 0) = -EQt;
    out.G_soc(0, 3) = T(-tanb * 3.0 * c.H / 4.0);
    out.G_soc(1, 3) = T(0);
    out.G_soc(2, 3) = T(0);
    out.h_soc = -EQt * pf.r;

    out.G_ort.template block<1, 3>(0, 0) = bx.transpose();
    out.G_ort(0, 3) = T(-c.H / 4.0);
    out.h_ort(0) = bx.dot(pf.r);
    return out;
}

// -------------------------------------------------------------------------
// Sphere : n_ort=0, n_soc=4, v=4
// -------------------------------------------------------------------------
template <typename T>
ProblemMats<0, 4, 4, T> problemMatrices(const Sphere& s, const TMat<4, 4, T>& g) {
    const TVec<3, T> p = g.template block<3, 1>(0, 3) + g.template block<3, 3>(0, 0) * s.r_offset.template cast<T>();

    ProblemMats<0, 4, 4, T> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = T(-s.R);
    out.G_soc.template block<3, 3>(1, 0) = -TMat<3, 3, T>::Identity();

    out.h_soc(0) = T(0);
    out.h_soc.template segment<3>(1) = -p;
    return out;
}

// -------------------------------------------------------------------------
// Ellipsoid : n_ort=0, n_soc=4, v=4  (x' P x <= 1, U = chol(P) upper)
// -------------------------------------------------------------------------
template <typename T>
ProblemMats<0, 4, 4, T> problemMatrices(const Ellipsoid& e, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T>(e, g);
    const TMat<3, 3, T> U = e.U.template cast<T>();
    const TMat<3, 3, T> UQt = U * pf.Q.transpose();

    ProblemMats<0, 4, 4, T> out;
    out.G_soc.setZero();
    out.G_soc(0, 3) = T(-1);
    out.G_soc.template block<3, 3>(1, 0) = -UQt;

    out.h_soc(0) = T(0);
    out.h_soc.template segment<3>(1) = -(UQt * pf.r);
    return out;
}

// -------------------------------------------------------------------------
// Polytope<NH> : n_ort=NH, n_soc=0, v=4  (A x <= b, locally)
// -------------------------------------------------------------------------
template <typename T, int NH>
ProblemMats<NH, 0, 4, T> problemMatrices(const Polytope<NH>& poly, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T, Polytope<NH>>(poly, g);
    const TMat<NH, 3, T> AQt = poly.A.template cast<T>() * pf.Q.transpose();

    ProblemMats<NH, 0, 4, T> out;
    out.G_ort.block(0, 0, NH, 3) = AQt;
    out.G_ort.block(0, 3, NH, 1) = -poly.b.template cast<T>();
    out.h_ort = AQt * pf.r;
    return out;
}

// -------------------------------------------------------------------------
// Polygon<NH> : n_ort=NH, n_soc=4, v=6 (extra: 2 local in-plane coordinates)
// -------------------------------------------------------------------------
template <typename T, int NH>
ProblemMats<NH, 4, 6, T> problemMatrices(const Polygon<NH>& poly, const TMat<4, 4, T>& g) {
    const auto pf = placeShape<T, Polygon<NH>>(poly, g);
    const TMat<3, 2, T> Qtilde = pf.Q.template block<3, 2>(0, 0);

    ProblemMats<NH, 4, 6, T> out;
    out.G_ort.setZero();
    out.G_ort.block(0, 3, NH, 1) = -poly.b.template cast<T>();
    out.G_ort.block(0, 4, NH, 2) = poly.A.template cast<T>();
    out.h_ort.setZero();

    out.G_soc.setZero();
    out.G_soc(0, 3) = T(-poly.R);
    out.G_soc.template block<3, 3>(1, 0) = -TMat<3, 3, T>::Identity();
    out.G_soc.template block<3, 2>(1, 4) = Qtilde;

    out.h_soc(0) = T(0);
    out.h_soc.template segment<3>(1) = -pf.r;
    return out;
}

} // namespace dcolpp::socp
