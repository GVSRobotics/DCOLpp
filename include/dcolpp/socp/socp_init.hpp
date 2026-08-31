#pragma once
// SocpInit construction for solveSocp -- cold and warm.
//
// The solver takes an optional init_hint (a full (x, s, z) triple). Two
// cold strategies, selected by SocpOptions::init_strategy:
//   Generic   -- solver.hpp's initializeSocp: unconstrained least-squares
//                fit of x, then bring2cone. No hint passed.
//   Geometric -- the DEFAULT (tested to converge in fewer iterations at no
//                loss of accuracy): seed the primal x from each shape's
//                inner/outer bounding-sphere radii. With r = center2 - center1,
//                  alpha_min = ||r|| / (r1_out + r2_out)
//                  alpha_max = ||r|| / (r1_in  + r2_in)
//                  alpha0    = sqrt(alpha_min * alpha_max)
//                  p0        = center1 + alpha0 * r1_out * r_hat
//                Only x is seeded; the dual z (a contact-force-like
//                multiplier, not a spatial point) has no geometric analogue
//                and is built separately (initializeSocpFromGuess).
//
// warmStartInit (further down) is the temporally-continuous path: seed from
// the previous converged solution at a nearby pose. See warm_start.hpp for
// the caller-side handle and pose metric.

#include "dcolpp/socp/problem_matrices.hpp"
#include "dcolpp/socp/solver.hpp"

namespace dcolpp::socp {

// Thin forwarder to the shape's cached bounding_sphere (computed once in
// its constructor, primitives.hpp). One generic template, every shape kind.
template <typename Shape>
inline const BoundingSphere& boundingSphere(const Shape& s) {
    return s.bounding_sphere;
}

// Per-shape extra decision-variable count, mirroring
// combineProblemMatrices' V1/V2 (V==4 <=> no extras). Specialized for the
// three shapes that carry one (Capsule/Cylinder: axial parameter) or two
// (Polygon: in-plane 2D coordinate).
template <typename Shape>
struct ExtraDim {
    static constexpr int value = 0;
};
template <>
struct ExtraDim<Capsule> {
    static constexpr int value = 1;
};
template <>
struct ExtraDim<Cylinder> {
    static constexpr int value = 1;
};
template <int NH>
struct ExtraDim<Polygon<NH>> {
    static constexpr int value = 2;
};

// Seed a shape's extra decision variables from the primal point p0, using
// the placed frame (R, r) = placeShape(shape, g).
//
// Capsule / Cylinder carry one extra t, the axial coordinate of the point
// on the shape's spine nearest p0 (problemMatrices constraints: SOC
// ||p - r - t*bx|| <= R*alpha, ORT |t| <= (L/2)*alpha). With bx = R*e_x
// the placed axis:
//   t = clamp( bx . (p0 - r),  -L/2,  L/2 )
inline Eigen::Matrix<double, 1, 1> extrasGuess(const Capsule& c, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);
    const double half_l = c.L / 2.0;
    const double t = std::max(-half_l, std::min(half_l, bx.dot(p0 - pf.r)));
    return Eigen::Matrix<double, 1, 1>(t);
}

inline Eigen::Matrix<double, 1, 1> extrasGuess(const Cylinder& c, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(c, g);
    const Eigen::Vector3d bx = pf.R * Eigen::Vector3d(1, 0, 0);
    const double half_l = c.L / 2.0;
    const double t = std::max(-half_l, std::min(half_l, bx.dot(p0 - pf.r)));
    return Eigen::Matrix<double, 1, 1>(t);
}

// Polygon carries two extras u, the in-plane 2D coordinate of p0 (SOC
// ||p - r - Rtilde*u|| <= R*alpha, ORT A*u <= alpha*b). With
// Rtilde = R[:, 0:2] the placed in-plane axes:
//   u = Rtilde^T (p0 - r)      (the out-of-plane component is dropped)
template <int NH>
Eigen::Matrix<double, 2, 1> extrasGuess(const Polygon<NH>& poly, const Eigen::Matrix4d& g, const Eigen::Vector3d& p0) {
    const auto pf = placeShape(poly, g);
    const Eigen::Matrix<double, 3, 2> Rtilde = pf.R.template block<3, 2>(0, 0);
    return Rtilde.transpose() * (p0 - pf.r);
}

// Builds the geometric [p;alpha;extras1;extras2] primal guess for the pair
// (shape1 @ identity, shape2 @ g), in the same column layout
// combineProblemMatrices uses.
template <typename Shape1, typename Shape2>
auto geometricPrimalGuess(const Shape1& shape1, const Shape2& shape2, const Eigen::Matrix4d& g) {
    constexpr int e1 = ExtraDim<Shape1>::value;
    constexpr int e2 = ExtraDim<Shape2>::value;
    constexpr int NX = 4 + e1 + e2;
    using XVec = Eigen::Matrix<double, NX, 1>;

    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    const auto pf1 = placeShape(shape1, I4);
    const auto pf2 = placeShape(shape2, g);
    const BoundingSphere b1 = boundingSphere(shape1);
    const BoundingSphere b2 = boundingSphere(shape2);

    const Eigen::Vector3d rvec = pf2.r - pf1.r;
    const double dist = rvec.norm();
    const Eigen::Vector3d rhat = (dist > 1e-9) ? Eigen::Vector3d(rvec / dist) : Eigen::Vector3d(1, 0, 0);

    const double eps = 1e-9;
    const double alpha_min = dist / std::max(b1.outer + b2.outer, eps);
    double alpha_max = dist / std::max(b1.inner + b2.inner, eps);
    if (alpha_max < alpha_min) alpha_max = 2.0 * alpha_min + eps; // degenerate/near-coincident centers
    const double alpha0 = std::sqrt(std::max(alpha_min, eps) * std::max(alpha_max, eps));

    const Eigen::Vector3d p0 = pf1.r + (alpha0 * b1.outer) * rhat;

    XVec x0;
    x0.template head<3>() = p0;
    x0(3) = alpha0;
    if constexpr (e1 > 0) x0.template segment<e1>(4) = extrasGuess(shape1, I4, p0);
    if constexpr (e2 > 0) x0.template segment<e2>(4 + e1) = extrasGuess(shape2, g, p0);
    return x0;
}

// Lift r by the smallest lambda*e that gives every SOC block a relative
// interior margin
//   (r0 - ||r_tail||) / r0 >= margin_frac
// and every ORT row a small absolute floor (ORT has no self-scale to be
// relative to). A relative SOC margin -- rather than bring2cone's fixed
// absolute push -- keeps a start that is near-exact in both s0 and z0 from
// producing a near-singular first NT scaling.
template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> pushToRelativeMargin(const StackVec<n_ort, n_soc1, n_soc2>& r, double margin_frac) {
    double lambda = 0.0;
    if constexpr (n_ort > 0) {
        const double min_ort = r.template head<n_ort>().minCoeff();
        lambda = std::max(lambda, 0.05 - min_ort); // ORT has no self-scale to be relative to; small absolute floor
    }
    if constexpr (n_soc1 > 0) {
        const Vec<n_soc1> r1 = r.template segment<n_soc1>(n_ort);
        const double tail_norm = r1.template tail<n_soc1 - 1>().norm();
        lambda = std::max(lambda, tail_norm / (1.0 - margin_frac) - r1(0));
    }
    if constexpr (n_soc2 > 0) {
        const Vec<n_soc2> r2 = r.template segment<n_soc2>(n_ort + n_soc1);
        const double tail_norm = r2.template tail<n_soc2 - 1>().norm();
        lambda = std::max(lambda, tail_norm / (1.0 - margin_frac) - r2(0));
    }
    if (lambda <= 0.0) return r;
    return StackVec<n_ort, n_soc1, n_soc2>(r + lambda * gen_e<n_ort, n_soc1, n_soc2>());
}
// Assembles a full SocpInit (primal = x0; slack and dual derived from it)
// for solveSocp's init_hint. x0 must be in the combined column layout
// (geometricPrimalGuess).
//
// s0: the solver residual is rz = s + Gx - h, so s = h - G*x0 (the sign
// matters -- G*x0 - h negates the SOC vector parts). Then pushed to a
// strict interior margin (a touching contact's true s lies on the SOC
// boundary; the first NT scaling needs an interior point).
//
// z0: complementary slackness s o z = 0 under the Jordan product
//   u o v = (u.v,  u0*v_tail + v0*u_tail)
// puts z, per SOC block, along the reflected ray (s_tail negated). Take
// that as a preferred direction z_pref (0 on ORT rows) and project it onto
// KKT stationarity {z : G^T z = -c}:
//   z0 = z_pref + G (G^T G)^{-1} (-c - G^T z_pref)
// which gives G^T z0 = -c exactly for any z_pref, reusing the (G^T G)^{-1}
// factor from x0's own fit. Then pushed to the same interior margin as s0.
template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> initializeSocpFromGuess(const DecisionVec<nx>& c,
                                                             const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                             const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                             const DecisionVec<nx>& x0) {
    using Z = StackVec<n_ort, n_soc1, n_soc2>;
    constexpr double kMarginFrac = 0.05;

    const Z s_tilde = h - G * x0;
    const Z s0 = pushToRelativeMargin<n_ort, n_soc1, n_soc2>(s_tilde, kMarginFrac);

    Z z_pref = Z::Zero(); // ORT rows assumed inactive: z_pref = 0 there
    if constexpr (n_soc1 > 0) {
        Vec<n_soc1> blk = s_tilde.template segment<n_soc1>(n_ort);
        blk.template tail<n_soc1 - 1>() *= -1.0;
        z_pref.template segment<n_soc1>(n_ort) = blk;
    }
    if constexpr (n_soc2 > 0) {
        Vec<n_soc2> blk = s_tilde.template segment<n_soc2>(n_ort + n_soc1);
        blk.template tail<n_soc2 - 1>() *= -1.0;
        z_pref.template segment<n_soc2>(n_ort + n_soc1) = blk;
    }

    SmallLLT<nx> F;
    F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(G));
    const DecisionVec<nx> w = F.solve(DecisionVec<nx>(-c - G.transpose() * z_pref));
    const Z z_exact = Z(z_pref + G * w);
    const Z z0 = pushToRelativeMargin<n_ort, n_soc1, n_soc2>(z_exact, kMarginFrac);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x0, s0, z0};
}

// Minimal re-centering for a warm start: lift r just into the interior --
// each SOC block to a soc_rel_margin relative margin, each ORT row to an
// ort_floor absolute floor. Unlike pushToRelativeMargin it does not lift
// inactive ORT rows uniformly, which would inflate s.z and pull the warm
// start's gap away from mu_prev.
template <int n_ort, int n_soc1, int n_soc2>
StackVec<n_ort, n_soc1, n_soc2> warmRecenter(const StackVec<n_ort, n_soc1, n_soc2>& r, double ort_floor,
                                             double soc_rel_margin) {
    StackVec<n_ort, n_soc1, n_soc2> out = r;
    if constexpr (n_ort > 0) {
        for (int i = 0; i < n_ort; ++i)
            if (out(i) < ort_floor) out(i) = ort_floor;
    }
    if constexpr (n_soc1 > 0) {
        const double t = out.template segment<n_soc1>(n_ort).template tail<n_soc1 - 1>().norm();
        const double need = t / (1.0 - soc_rel_margin) - out(n_ort);
        if (need > 0.0) out(n_ort) += need;
    }
    if constexpr (n_soc2 > 0) {
        const int b = n_ort + n_soc1;
        const double t = out.template segment<n_soc2>(b).template tail<n_soc2 - 1>().norm();
        const double need = t / (1.0 - soc_rel_margin) - out(b);
        if (need > 0.0) out(b) += need;
    }
    return out;
}

// Warm-start init for temporally-continuous queries (a physics step, a
// traj-opt inner loop, a smooth sweep): seed from the previous converged
// solution at a nearby pose instead of the cold geometric guess.
// pose_move (warm_start.hpp::poseMoveMetric) selects a tier:
//
//  * pose_move ~ 0 (body at rest / fixed-base grasp): the previous
//    (x*, s*, z*) is still a KKT point to pdip_tol. Feed it back raw, with
//    only an absolute-floor nudge for strict interiority (raw converged
//    values can make the first NT scaling singular). No projection, no
//    Cholesky -- solveSocp accepts it at iteration 1.
//
//  * pose_move small but nonzero: rebuild the slack from the new data,
//    s0 = h_new - G_new*x_prev, and re-project z_prev onto stationarity
//    {z : G_new^T z = -c} for the new G (same (G^T G)^{-1} projection as
//    initializeSocpFromGuess) -- otherwise the solver can drive mu below
//    pdip_tol while z's magnitude is still off. Both then get warmRecenter.
//
// The caller falls back to a cold solve if this hint fails to converge in
// a reduced iteration cap or converges with lagging KKT residuals.
template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> warmStartInit(const DecisionVec<nx>& c,
                                                   const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                   const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                   const DecisionVec<nx>& x_prev,
                                                   const StackVec<n_ort, n_soc1, n_soc2>& z_prev, double pose_move) {
    using Z = StackVec<n_ort, n_soc1, n_soc2>;
    constexpr double kTinyMove = 1e-6;

    if (pose_move < kTinyMove) {
        // Same problem to pdip_tol -- reuse (x*, s*, z*) essentially raw.
        const Z s0 = warmRecenter<n_ort, n_soc1, n_soc2>(Z(h - G * x_prev), 1e-12, 1e-9);
        const Z z0 = warmRecenter<n_ort, n_soc1, n_soc2>(z_prev, 1e-12, 1e-9);
        return SocpInit<n_ort, n_soc1, n_soc2, nx>{x_prev, s0, z0};
    }

    const Z s0 = warmRecenter<n_ort, n_soc1, n_soc2>(Z(h - G * x_prev), 1e-7, 1e-3);

    SmallLLT<nx> F;
    F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(G));
    const DecisionVec<nx> w = F.solve(DecisionVec<nx>(-c - G.transpose() * z_prev));
    const Z z0 = warmRecenter<n_ort, n_soc1, n_soc2>(Z(z_prev + G * w), 1e-7, 1e-3);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x_prev, s0, z0};
}

// Central-path-preserving warm-start init (for proximityContactJacobian,
// whose frozen Hessian needs s o z proportional to e, not merely
// mu < pdip_tol). Carries the previous converged (s_prev, z_prev) -- already
// on the central path for the old G -- forward with only the exact update
// for the new G:
//   s0 = h_new - G_new * x_prev                     (exact new slack)
//   z0 = z_prev + G (G^T G)^{-1} (-c - G^T z_prev)   (min. re-projection onto
//                                                    G^T z = -c)
// then a single UNIFORM lambda*e lift to margin `interior_margin` if either
// pokes outside (liftToMargin -- NOT warmRecenter, whose per-block lift
// would make s0 o z0 non-uniform and cost the solver re-centering
// iterations). An unmoved pose returns the previous point essentially raw,
// so solveSocp converges in 1 iteration at the same pdip_tol as cold --
// same stopping rule, no early-exit trick.
template <int n_ort, int n_soc1, int n_soc2, int nx>
SocpInit<n_ort, n_soc1, n_soc2, nx> warmStartInitCentral(const DecisionVec<nx>& c,
                                                          const ConstraintMat<n_ort, n_soc1, n_soc2, nx>& G,
                                                          const StackVec<n_ort, n_soc1, n_soc2>& h,
                                                          const DecisionVec<nx>& x_prev,
                                                          const StackVec<n_ort, n_soc1, n_soc2>& z_prev,
                                                          double interior_margin = 1e-9) {
    using Z = StackVec<n_ort, n_soc1, n_soc2>;

    const Z s0 = liftToMargin<n_ort, n_soc1, n_soc2>(Z(h - G * x_prev), interior_margin);

    SmallLLT<nx> F;
    F.compute(gramLower<n_ort + n_soc1 + n_soc2, nx>(G));
    const DecisionVec<nx> w = F.solve(DecisionVec<nx>(-c - G.transpose() * z_prev));
    const Z z0 = liftToMargin<n_ort, n_soc1, n_soc2>(Z(z_prev + G * w), interior_margin);

    return SocpInit<n_ort, n_soc1, n_soc2, nx>{x_prev, s0, z0};
}

} // namespace dcolpp::socp
