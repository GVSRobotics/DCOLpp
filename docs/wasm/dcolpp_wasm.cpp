// docs/wasm/dcolpp_wasm.cpp
//
// Emscripten/embind bindings exposing the REAL DCOL++ solver
// (dcolpp::socp::proximityContactJacobian, the same function
// tests/test_contact_manifold.cpp exercises) to the interactive demo on
// docs/index.html. Mirrors the structure of the sibling iDCOL project's
// docs/wasm/idcol_wasm.cpp: a small fixed shape registry, a pose-builder,
// and a single "solve this pair at this pose" entry point returning a JS
// object -- no reimplementation of the solver in JS.
//
// Seven shapes, one representative example per DCOL++ primitive (indices
// 0-6). DCOL++ has no type-erased shape value-type (each shape is its own
// C++ type, proximityContactJacobian is templated on both), so unlike
// iDCOL's uniform ShapeSpec registry this needs an explicit double
// dispatch (std::variant + nested std::visit) over the 7x7 = 49 concrete
// (Shape1,Shape2) instantiations.
//
// Build (from docs/wasm/, after activating emsdk):
//   em++ dcolpp_wasm.cpp ../../src/se3.cpp ../../src/socp_analytic_derivatives.cpp \
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

// ---------------------------------------------------------------------
// Shape registry (indices 0-6): one representative example per DCOL++
// primitive, roughly unit scale (matches this project's own tests, e.g.
// tests/test_contact_degeneracy.cpp's makeBox for the polytope).
// ---------------------------------------------------------------------
Polytope<6> makeCube(double h) {
    Eigen::Matrix<double, 6, 3> A;
    Eigen::Matrix<double, 6, 1> b;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    b.setConstant(h);
    return Polytope<6>(A, b);
}

Polygon<4> makeSquare(double h, double R) {
    Eigen::Matrix<double, 4, 2> A;
    Eigen::Matrix<double, 4, 1> b;
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    b.setConstant(h);
    return Polygon<4>(A, b, R);
}

using ShapeVariant = std::variant<Sphere, Capsule, Cylinder, Cone, Ellipsoid, Polytope<6>, Polygon<4>>;

ShapeVariant buildShape(int key) {
    switch (key) {
        case 0: return Sphere(0.8);
        case 1: return Capsule(0.3, 1.2);
        case 2: return Cylinder(0.5, 1.4);
        case 3: return Cone(1.3, 22.0 * kPi / 180.0);
        case 4: {
            Matrix3d P = Vector3d(1.0 / (0.9 * 0.9), 1.0 / (0.6 * 0.6), 1.0 / (0.5 * 0.5)).asDiagonal();
            return Ellipsoid(P);
        }
        case 5: return makeCube(0.5);
        case 6:
        default: return makeSquare(0.4, 0.15);
    }
}

const ShapeVariant& getShape(int key) {
    static const ShapeVariant shapes[7] = {buildShape(0), buildShape(1), buildShape(2), buildShape(3),
                                            buildShape(4), buildShape(5), buildShape(6)};
    return shapes[key < 0 || key > 6 ? 0 : key];
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
// so the (keyA,keyB) -> (Shape1,Shape2) resolution has to happen via
// nested std::visit -- this instantiates all 7x7=49 combinations.
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
// JS-facing API
// ---------------------------------------------------------------------
val solvePair(int keyA, int keyB, double rx, double ry, double rz, double r00, double r01, double r02, double r10,
              double r11, double r12, double r20, double r21, double r22) {
    const Matrix4d g = makeG(rx, ry, rz, r00, r01, r02, r10, r11, r12, r20, r21, r22);
    return solvePairImpl(getShape(keyA), getShape(keyB), g);
}

// Per-shape parameters, for the JS-side wireframe renderer (so shape
// dimensions live in exactly one place: this file's buildShape above).
val getShapeParamsJs(int key) {
    val out = val::object();
    std::visit(
        [&](const auto& s) {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, Sphere>) {
                out.set("kind", std::string("sphere"));
                out.set("R", s.R);
            } else if constexpr (std::is_same_v<S, Capsule>) {
                out.set("kind", std::string("capsule"));
                out.set("R", s.R);
                out.set("L", s.L);
            } else if constexpr (std::is_same_v<S, Cylinder>) {
                out.set("kind", std::string("cylinder"));
                out.set("R", s.R);
                out.set("L", s.L);
            } else if constexpr (std::is_same_v<S, Cone>) {
                out.set("kind", std::string("cone"));
                out.set("H", s.H);
                out.set("beta", s.beta);
            } else if constexpr (std::is_same_v<S, Ellipsoid>) {
                out.set("kind", std::string("ellipsoid"));
                out.set("a", 1.0 / std::sqrt(s.P(0, 0)));
                out.set("b", 1.0 / std::sqrt(s.P(1, 1)));
                out.set("c", 1.0 / std::sqrt(s.P(2, 2)));
            } else if constexpr (std::is_same_v<S, Polytope<6>>) {
                out.set("kind", std::string("polytope"));
                val A = val::array(), b = val::array();
                for (int i = 0; i < 6; ++i) {
                    A.call<void>("push", val::array(std::vector<double>{s.A(i, 0), s.A(i, 1), s.A(i, 2)}));
                    b.call<void>("push", s.b(i));
                }
                out.set("A", A);
                out.set("b", b);
            } else if constexpr (std::is_same_v<S, Polygon<4>>) {
                out.set("kind", std::string("polygon"));
                val A = val::array(), b = val::array();
                for (int i = 0; i < 4; ++i) {
                    A.call<void>("push", val::array(std::vector<double>{s.A(i, 0), s.A(i, 1)}));
                    b.call<void>("push", s.b(i));
                }
                out.set("A", A);
                out.set("b", b);
                out.set("R", s.R);
            }
        },
        getShape(key));
    return out;
}

} // namespace

EMSCRIPTEN_BINDINGS(dcolpp_module) {
    function("solvePair", &solvePair);
    function("getShapeParams", &getShapeParamsJs);
}
