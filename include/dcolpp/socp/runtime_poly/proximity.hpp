#pragma once
// dcolpp::socp::runtime_poly -- proximity queries for the pure-ORT PolytopeX pairs
// (PolytopeX-PolytopeX and Plane-PolytopeX): minimum uniform scaling alpha
// (< 1 penetrating, == 1 touching, > 1 apart), the witness point, and their
// xi-Jacobians. Mirrors proximity.hpp. Each takes an optional
// ContactWarmStateX* for temporally-continuous callers; nullptr is the cold
// path.

#include <Eigen/Dense>

#include "dcolpp/socp/runtime_poly/analytic_derivatives.hpp"
#include "dcolpp/socp/runtime_poly/problem_matrices.hpp"
#include "dcolpp/socp/runtime_poly/solver.hpp"
#include "dcolpp/socp/runtime_poly/warm_start.hpp"

namespace dcolpp::socp::runtime_poly {

struct ProximityResultX {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero(); // reference (shape-1) frame
    int iters = 0;
    bool converged = false;
    bool plane_flipped = false;
};

struct AlphaGradientResultX {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 1, 6> grad = Eigen::Matrix<double, 1, 6>::Zero(); // d(alpha)/dxi, xi = [w; v]
    int iters = 0;
    bool converged = false;
    bool plane_flipped = false;
};

struct ProximityJacobianResultX {
    double alpha = 0.0;
    Eigen::Vector3d witness_point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 4, 6> jacobian = Eigen::Matrix<double, 4, 6>::Zero(); // rows [wx,wy,wz,alpha], cols xi=[w;v]
    int iters = 0;
    bool converged = false;
    bool plane_flipped = false;
};

// --- PolytopeX vs PolytopeX ------------------------------------------------

inline ProximityResultX proximity(const PolytopeX& shape1, const PolytopeX& shape2, const Eigen::Matrix4d& g,
                                  const SocpOptions& opt = SocpOptions{}, ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    const CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocp(shape1.bounding_sphere, shape2.bounding_sphere, g, combined.c, combined.G,
                                        combined.h, opt); });
    ProximityResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    return res;
}

inline AlphaGradientResultX alphaGradient(const PolytopeX& shape1, const PolytopeX& shape2, const Eigen::Matrix4d& g,
                                          const SocpOptions& opt = SocpOptions{}, ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    const CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocp(shape1.bounding_sphere, shape2.bounding_sphere, g, combined.c, combined.G,
                                        combined.h, opt); });
    AlphaGradientResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged)
        res.grad = computeProximityGradient(shape2, g, sol.x, sol.z, combined.n_ort1);
    return res;
}

inline ProximityJacobianResultX proximityJacobian(const PolytopeX& shape1, const PolytopeX& shape2,
                                                  const Eigen::Matrix4d& g, const SocpOptions& opt = SocpOptions{},
                                                  ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(shape1, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    const CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocp(shape1.bounding_sphere, shape2.bounding_sphere, g, combined.c, combined.G,
                                        combined.h, opt); });
    ProximityJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    if (sol.converged)
        res.jacobian = diffSocp(shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1);
    return res;
}

// --- Plane (shape 1) vs PolytopeX (shape 2) ------------------------------

inline ProximityResultX proximity(const Plane& plane, const PolytopeX& shape2, const Eigen::Matrix4d& g,
                                  const SocpOptions& opt = SocpOptions{}, ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(plane, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const bool flipped = applyPlaneFlipX(plane, g, combined.G, combined.h);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocpPlane(plane.normal, plane.d, shape2.bounding_sphere, g, combined.c,
                                             combined.G, combined.h, opt); });
    ProximityResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    res.plane_flipped = flipped;
    return res;
}

inline ProximityJacobianResultX proximityJacobian(const Plane& plane, const PolytopeX& shape2,
                                                  const Eigen::Matrix4d& g, const SocpOptions& opt = SocpOptions{},
                                                  ContactWarmStateX* warm = nullptr) {
    ProblemMatsX P1_local;
    const ProblemMatsX& P1 = cachedBody1Matrices(plane, warm, P1_local);
    const ProblemMatsX P2 = problemMatrices(shape2, g);
    CombinedProblemX combined = combineProblemMatrices(P1, P2);
    const bool flipped = applyPlaneFlipX(plane, g, combined.G, combined.h);
    const SocpResultX sol = solveForQuery(
        combined.c, combined.G, combined.h, opt, warm, WarmSeed::Standard, WarmStartConfig::kResidTolMul, g,
        [&] { return solveProximitySocpPlane(plane.normal, plane.d, shape2.bounding_sphere, g, combined.c,
                                             combined.G, combined.h, opt); });
    ProximityJacobianResultX res;
    res.alpha = sol.x(3);
    res.witness_point = sol.x.head<3>();
    res.iters = sol.iters;
    res.converged = sol.converged;
    res.plane_flipped = flipped;
    if (sol.converged)
        res.jacobian = diffSocp(shape2, g, sol.x, sol.s, sol.z, combined.G, combined.n_ort1);
    return res;
}

} // namespace dcolpp::socp::runtime_poly
