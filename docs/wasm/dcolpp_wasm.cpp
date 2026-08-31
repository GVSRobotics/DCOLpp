// docs/wasm/dcolpp_wasm.cpp
//
// Emscripten/embind bindings exposing the REAL DCOL++ solver
// (dcolpp::socp::proximityContactJacobian, the same function
// tests/test_contact_manifold.cpp exercises) to the interactive demo on
// docs/index.html: one "solve this pair at this pose" entry point that
// takes two shape SPEC objects built JS-side, materialises them, and
// returns a JS result object -- no reimplementation of the solver in JS,
// and all shape geometry (defaults + randomisation) lives in the page.
//
// Eight primitives. DCOL++ has no type-erased shape value-type (each shape
// is its own C++ type, proximityContactJacobian is templated on both), so
// this needs an explicit double dispatch (std::variant + nested std::visit)
// over the 8x8 = 64 concrete (Shape1,Shape2) instantiations. Polytope and
// Polygon are pinned to NHMAX half-planes; JS pads shorter face lists.
//
// Build (from docs/wasm/, after activating emsdk):
//   em++ dcolpp_wasm.cpp ../../src/se3.cpp ../../src/analytic_derivatives.cpp \
//        -I../../include -I<eigen include dir> -std=c++17 -lembind -O3 \
//        -s MODULARIZE=1 -s EXPORT_NAME=DcolppModule -s ENVIRONMENT=web \
//        -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 \
//        -o dcolpp_wasm.js

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <Eigen/Dense>
#include <variant>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/proximity_contact.hpp"

using namespace dcolpp::socp;
using namespace emscripten;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace {
constexpr double kPi = 3.14159265358979323846;

// Fixed max half-plane count for Polytope/Polygon. Fewer-face shapes are
// represented by padding unused rows with a far-away redundant half-plane
// (done JS-side) that the solver never activates -- so one template
// instantiation covers every face count 3..16 without binary bloat.
constexpr int NHMAX = 16;

// ---------------------------------------------------------------------
// Shape construction from a JS spec object. All geometry now lives in
// docs/index.html (DEFAULTS + randomShape); this file just materialises
// whatever spec it is handed. Spec shapes:
//   {kind:"sphere",   R}
//   {kind:"capsule",  R, L}   {kind:"cylinder", R, L}
//   {kind:"cone",     H, beta}
//   {kind:"truncatedcone", R_bottom, R_top, L}
//   {kind:"ellipsoid", a, b, c}
//   {kind:"polytope", A:[[x,y,z] x NHMAX], b:[NHMAX]}
//   {kind:"polygon",  A:[[x,y]   x NHMAX], b:[NHMAX], R}
// ---------------------------------------------------------------------
using ShapeVariant =
    std::variant<Sphere, Capsule, Cylinder, Cone, TruncatedCone, Ellipsoid, Polytope<NHMAX>, Polygon<NHMAX>>;

ShapeVariant buildShape(const val& s) {
    const std::string kind = s["kind"].as<std::string>();
    if (kind == "sphere") return Sphere(s["R"].as<double>());
    if (kind == "capsule") return Capsule(s["R"].as<double>(), s["L"].as<double>());
    if (kind == "cylinder") return Cylinder(s["R"].as<double>(), s["L"].as<double>());
    if (kind == "cone") return Cone(s["H"].as<double>(), s["beta"].as<double>());
    if (kind == "truncatedcone")
        return TruncatedCone(s["R_bottom"].as<double>(), s["R_top"].as<double>(), s["L"].as<double>());
    if (kind == "ellipsoid")
        return Ellipsoid(s["a"].as<double>(), s["b"].as<double>(), s["c"].as<double>());
    if (kind == "polytope") {
        Eigen::Matrix<double, NHMAX, 3> A;
        Eigen::Matrix<double, NHMAX, 1> b;
        const val Aj = s["A"], bj = s["b"];
        for (int i = 0; i < NHMAX; ++i) {
            const val row = Aj[i];
            A(i, 0) = row[0].as<double>();
            A(i, 1) = row[1].as<double>();
            A(i, 2) = row[2].as<double>();
            b(i) = bj[i].as<double>();
        }
        return Polytope<NHMAX>(A, b);
    }
    // polygon
    Eigen::Matrix<double, NHMAX, 2> A;
    Eigen::Matrix<double, NHMAX, 1> b;
    const val Aj = s["A"], bj = s["b"];
    for (int i = 0; i < NHMAX; ++i) {
        const val row = Aj[i];
        A(i, 0) = row[0].as<double>();
        A(i, 1) = row[1].as<double>();
        b(i) = bj[i].as<double>();
    }
    return Polygon<NHMAX>(A, b, s["R"].as<double>());
}

Matrix4d makeG(double rx, double ry, double rz, double r00, double r01, double r02, double r10, double r11,
               double r12, double r20, double r21, double r22) {
    Matrix4d g = Matrix4d::Identity();
    g(0, 0) = r00; g(0, 1) = r01; g(0, 2) = r02;
    g(1, 0) = r10; g(1, 1) = r11; g(1, 2) = r12;
    g(2, 0) = r20; g(2, 1) = r21; g(2, 2) = r22;
    g(0, 3) = rx;  g(1, 3) = ry;  g(2, 3) = rz;
    return g;
}

val packResult(const ProximityContactJacobianResult& r) {
    val res = val::object();
    res.set("converged", r.converged);
    if (!r.converged) return res;

    res.set("alpha", r.alpha);
    res.set("witness", val::array(std::vector<double>{r.witness_point(0), r.witness_point(1), r.witness_point(2)}));
    res.set("normal", val::array(std::vector<double>{r.normal(0), r.normal(1), r.normal(2)}));

    // Analytic Jacobians w.r.t. shape 2's local twist xi = [w; v] (rotation
    // first, translation second), perturbing as g -> g * se3::Exp(xi) --
    // flat, ROW-MAJOR. jacobian: 4x6, rows [wx,wy,wz,alpha]. normalJacobian:
    // 3x6, rows the unit normal's [nx,ny,nz]. Both at the single witness
    // point x*; see the demo's "Perturb g & watch" overlay for what they
    // predict (x* + J*xi) vs. the re-solved truth.
    val jac = val::array();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 6; ++j) jac.call<void>("push", r.jacobian(i, j));
    res.set("jacobian", jac);
    val njac = val::array();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 6; ++j) njac.call<void>("push", r.normal_jacobian(i, j));
    res.set("normalJacobian", njac);

    res.set("contactManifoldDim", r.contact_manifold_dim);
    res.set("normalConeDim", r.normal_cone_dim);
    res.set("witnessJacobianValid", r.witness_jacobian_valid);
    res.set("normalJacobianValid", r.normal_jacobian_valid);

    val pts = val::array();
    for (size_t i = 0; i < r.contact_manifold_points.size(); ++i) {
        const auto& p = r.contact_manifold_points[i];
        pts.call<void>("push", val::array(std::vector<double>{p(0), p(1), p(2)}));
    }
    res.set("manifoldPoints", pts);
    return res;
}

// Double dispatch: proximityContactJacobian is templated on both shapes,
// so the (specA,specB) -> (Shape1,Shape2) resolution has to happen via
// nested std::visit -- this instantiates all 8x8=64 combinations.
val solvePairImpl(const ShapeVariant& shapeA, const ShapeVariant& shapeB, const Matrix4d& g) {
    SocpOptions opt;
    opt.compute_contact_manifold = true; // implies compute_degeneracy_info's fields too
    opt.contact_manifold_points = 4;

    ProximityContactJacobianResult r;
    std::visit(
        [&](const auto& s1) {
            std::visit([&](const auto& s2) { r = proximityContactJacobian(s1, s2, g, opt); }, shapeB);
        },
        shapeA);
    return packResult(r);
}

// ---------------------------------------------------------------------
// JS-facing API: two shape spec objects (see buildShape) + body B's pose
// as translation + row-major rotation.
// ---------------------------------------------------------------------
val solvePair(val specA, val specB, double rx, double ry, double rz, double r00, double r01, double r02, double r10,
              double r11, double r12, double r20, double r21, double r22) {
    const Matrix4d g = makeG(rx, ry, rz, r00, r01, r02, r10, r11, r12, r20, r21, r22);
    return solvePairImpl(buildShape(specA), buildShape(specB), g);
}

} // namespace

EMSCRIPTEN_BINDINGS(dcolpp_module) {
    function("solvePair", &solvePair);
}
