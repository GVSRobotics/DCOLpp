// Per-shape chain-rule derivatives and H_frozen. Declarations + overview
// in dcolpp/socp/analytic_derivatives.hpp.
#include "dcolpp/socp/analytic_derivatives.hpp"

namespace dcolpp::socp {

ShapeXiDerivative<0, 4, 4> sphereXiDerivative(const Sphere& /*shape*/, const Eigen::Matrix4d& g0) {
    // Stage 1+2 (generic): d(center r = pose origin)/dxi.
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());

    // Stage 3: h_soc = (0; -r), d(h_soc)/dr = -I (the rest of G_soc, h_soc(0)
    // is constant, independent of r or R). Row 0 stays at the default zero.
    ShapeXiDerivative<0, 4, 4> out;
    out.dHsoc.block<3, 6>(1, 0) = -dr_dxi;
    return out;
}

ShapeXiDerivative<2, 4, 5> capsuleXiDerivative(const Capsule& /*shape*/, const Eigen::Matrix4d& g0, double t,
                                                const Eigen::Vector4d& z_soc) {
    // Stage 1+2 (generic): d(center r)/dxi and d(axis bx = R*e1)/dxi -- bx is
    // a direction, so dRotatedVectorDXi with the local axis e1, not dPointDXi.
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    // Stage 3: x_local=(p,alpha,t), z_soc=(z0,z1,z2,z3) frozen -- only R and r
    // vary with xi.
    ShapeXiDerivative<2, 4, 5> out;

    // G_soc*x_local = (-R_cap*alpha; -p + bx*t): only the bx*t term
    // (t frozen) carries xi-dependence, via bx alone. Chain rule:
    // d(G_soc*x_local)/dxi = t * d(bx)/dxi.
    out.dGsocX.block<3, 6>(1, 0) = t * dbx_dxi;

    // h_soc = (0; -r): d(h_soc)/dr = -I, chain rule as in sphereXiDerivative.
    out.dHsoc.block<3, 6>(1, 0) = -dr_dxi;

    // (G_soc^T*z_soc)_4 = bx . z_soc's spatial part -- the only R-dependent
    // entry. Chain rule: d(bx . z_vec)/dxi = z_vec . d(bx)/dxi, z_vec frozen.
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    out.dGsocTz.row(4) = z_vec.transpose() * dbx_dxi;

    // G_ort, h_ort are both constant (no R or r dependence) -- their
    // derivatives, and dGortTz, stay at the struct's default zero.
    return out;
}

ShapeXiDerivative<4, 4, 5> cylinderXiDerivative(const Cylinder& /*shape*/, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                 const Eigen::Vector4d& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx = R0 * Eigen::Vector3d::UnitX();
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    ShapeXiDerivative<4, 4, 5> out;

    // G_soc: identical structure to Capsule.
    out.dGsocX.block<3, 6>(1, 0) = t * dbx_dxi;
    out.dHsoc.block<3, 6>(1, 0) = -dr_dxi;
    const Eigen::Vector3d z_soc_vec = z_soc.tail<3>();
    out.dGsocTz.row(4) = z_soc_vec.transpose() * dbx_dxi;

    // G_ort rows 0,1 (the axial clamp on t) are constant -- stay zero. Rows
    // 2,3 clip the disk faces: G_ort(2,:)*x = -bx.p - L/2*alpha,
    // G_ort(3,:)*x = bx.p - L/2*alpha -- only the bx.p term (p frozen)
    // carries xi-dependence.
    out.dGortX.row(2) = -(p.transpose() * dbx_dxi);
    out.dGortX.row(3) = p.transpose() * dbx_dxi;

    // h_ort = (0,0,-bx.r,bx.r): both bx and r vary with xi -- product rule.
    const Eigen::Matrix<double, 1, 6> d_bxdotr = r.transpose() * dbx_dxi + bx.transpose() * dr_dxi;
    out.dHort.row(2) = -d_bxdotr;
    out.dHort.row(3) = d_bxdotr;

    // (G_ort^T*z_ort)'s p-columns (0-2) = (z_ort(3)-z_ort(2)) * bx; its
    // alpha- and t-columns pick out constant coefficients only.
    const double z_diff = z_ort(3) - z_ort(2);
    out.dGortTz.block<3, 6>(0, 0) = z_diff * dbx_dxi;

    return out;
}

ShapeXiDerivative<1, 3, 4> coneXiDerivative(const Cone& shape, const Eigen::Matrix4d& g0, const Eigen::Vector3d& p,
                                             double z_ort, const Eigen::Vector3d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx0 = R0 * Eigen::Vector3d::UnitX();

    const double tanb = std::tan(shape.beta);
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    ShapeXiDerivative<1, 3, 4> out;

    // G_soc*x_local = -E*R^T*p + (const)*alpha: only E*R^T*p carries
    // xi-dependence -- p is a frozen decision variable (no product rule
    // needed, unlike h_soc below).
    out.dGsocX = -E * se3::dInverseRotatedVectorDXi(g0, p);

    // h_soc = -E*R^T*r: r = r(xi) also varies with xi, so R^T*r needs the
    // ordinary product rule -- one term treats R^T as frozen at R0^T
    // (giving R0^T*dr_dxi), the other treats r as frozen at its current
    // value r0 (giving dInverseRotatedVectorDXi evaluated there, not at a
    // decision variable).
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc = -E * dRtr_dxi;

    // (G_soc^T*z_soc)'s first 3 entries = -R*(E*z_soc) -- E, z_soc are
    // xi-independent, so this is a single dRotatedVectorDXi call on that
    // fixed local vector.
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, Eigen::Vector3d(E * z_soc));

    // G_ort*x_local = bx.p - H/4*alpha: only bx.p carries xi-dependence
    // (p frozen).
    out.dGortX = p.transpose() * dbx_dxi;

    // h_ort = bx.r: both vary with xi -- product rule, same pattern as
    // Cylinder's h_ort.
    out.dHort = r0.transpose() * dbx_dxi + bx0.transpose() * dr_dxi;

    // (G_ort^T*z_ort)'s p-columns (0-2) = z_ort*bx (z_ort a scalar here,
    // n_ort=1); its alpha-column picks out a constant coefficient only.
    out.dGortTz.block<3, 6>(0, 0) = z_ort * dbx_dxi;

    return out;
}

// Frustum = Cone's SOC block verbatim + Cylinder's +-bx end-cap orthant
// rows (here rows 0/1 instead of Cylinder's 2/3). Only the SOC alpha
// constant differs from Cone (apex_dist vs 3H/4) and it's constant, so it
// contributes nothing to any derivative -- the SOC code below is a
// character-for-character copy of coneXiDerivative's.
ShapeXiDerivative<2, 3, 4> truncatedConeXiDerivative(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                      const Eigen::Vector3d& p, const Eigen::Vector2d& z_ort,
                                                      const Eigen::Vector3d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx0 = R0 * Eigen::Vector3d::UnitX();

    const double tanb = shape.tan_beta;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    ShapeXiDerivative<2, 3, 4> out;

    // --- SOC block: identical to Cone (see coneXiDerivative for the notes) ---
    out.dGsocX = -E * se3::dInverseRotatedVectorDXi(g0, p);
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc = -E * dRtr_dxi;
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, Eigen::Vector3d(E * z_soc));

    // --- ORT block: rows 0/1 clip the flat caps. G_ort(0,:)*x = bx.p - L/2*alpha,
    // G_ort(1,:)*x = -bx.p - L/2*alpha -- only the bx.p term (p frozen)
    // carries xi-dependence. h_ort = (bx.r, -bx.r): product rule. Same shape
    // as Cylinder's rows 2/3. ---
    const Eigen::Matrix<double, 1, 6> dbxp = p.transpose() * dbx_dxi;
    out.dGortX.row(0) = dbxp;
    out.dGortX.row(1) = -dbxp;

    const Eigen::Matrix<double, 1, 6> d_bxdotr = r0.transpose() * dbx_dxi + bx0.transpose() * dr_dxi;
    out.dHort.row(0) = d_bxdotr;
    out.dHort.row(1) = -d_bxdotr;

    // (G_ort^T*z_ort)'s p-columns (0-2) = (z_ort(0) - z_ort(1)) * bx.
    out.dGortTz.block<3, 6>(0, 0) = (z_ort(0) - z_ort(1)) * dbx_dxi;

    return out;
}

ShapeXiDerivative<0, 4, 4> ellipsoidXiDerivative(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                  const Eigen::Vector3d& p, const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3);
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, Eigen::Vector3d::Zero());

    ShapeXiDerivative<0, 4, 4> out;

    // G_soc*x_local = (-alpha; -U*R^T*p): only U*R^T*p carries xi-dependence,
    // via R^T*p -- p frozen, no product rule (same pattern as Cone's dGsocX,
    // U standing in for E).
    out.dGsocX.block<3, 6>(1, 0) = -shape.U * se3::dInverseRotatedVectorDXi(g0, p);

    // h_soc = (0; -U*R^T*r): r = r(xi) also varies -- product rule, same
    // pattern as Cone's h_soc.
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc.block<3, 6>(1, 0) = -shape.U * dRtr_dxi;

    // (G_soc^T*z_soc)'s first 3 entries = -R*(U^T*z_vec) -- U, z_vec are
    // xi-independent.
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, Eigen::Vector3d(shape.U.transpose() * z_vec));

    return out;
}

// H_frozen (6x6) = d/dxi[ grad(xi) ] with x, z frozen at their converged
// values (grad = -q^T z; see computeProximityGradient). Assembled in closed
// form from the stk* helpers in the header -- each the once-more pose
// derivative of "frozen covector times one SE(3) field-Jacobian" -- plus,
// for the shapes whose first derivative already needed a product rule, the
// constant outer products of two first-derivative Jacobians. No e_j loop.

Eigen::Matrix<double, 6, 6> sphereHessianFrozen(const Sphere& /*shape*/, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    // grad's only xi-dependent factor is z_vec^T d(center = pose origin)/dxi.
    return stkPoint(R0, z_vec, Eigen::Vector3d::Zero());
}

Eigen::Matrix<double, 6, 6> capsuleHessianFrozen(const Capsule& /*shape*/, const Eigen::Matrix4d& g0, double t,
                                                  const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    // SOC only: z_vec^T ( d(center)/dxi + t d(R e1)/dxi ).
    return stkPoint(R0, z_vec, Eigen::Vector3d::Zero()) + t * stkRot(R0, z_vec, Eigen::Vector3d::UnitX());
}

Eigen::Matrix<double, 6, 6> cylinderHessianFrozen(const Cylinder& /*shape*/, const Eigen::Matrix4d& g0,
                                                   const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                   const Eigen::Vector4d& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx0 = R0 * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d z_vec = z_soc.tail<3>();

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    // SOC block: same shape as Capsule.
    const Eigen::Matrix<double, 6, 6> dS_soc =
        -(stkPoint(R0, z_vec, Eigen::Vector3d::Zero()) + t * stkRot(R0, z_vec, Eigen::Vector3d::UnitX()));

    // ORT rows 2,3: q-row is +-( bx.(r - p) ) plus constant alpha terms.
    // d/dxi of ( dbx_dxi^T (r - p) + bx^T dr_dxi ) = two stk* terms + the two
    // constant first-Jacobian outer products. row 3's q = -row 2's q.
    const Eigen::Matrix<double, 6, 6> BR = dbx_g0.transpose() * dr_g0 + dr_g0.transpose() * dbx_g0 +
                                           stkRot(R0, r0, Eigen::Vector3d::UnitX()) +
                                           stkPoint(R0, bx0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 6, 6> dq_row2 = -BR + stkRot(R0, p, Eigen::Vector3d::UnitX());
    const Eigen::Matrix<double, 6, 6> dS_ort = (z_ort(2) - z_ort(3)) * dq_row2;

    return -(dS_soc + dS_ort);
}

// ORT block (Cone/Frustum share it): q-row is bx.(r - p) plus constant alpha
// terms. d/dxi of ( dbx_dxi^T (r - p) + bx^T dr_dxi ) -> two stk* terms + the
// two constant first-Jacobian outer products.
static Eigen::Matrix<double, 6, 6> coneOrtQSecondDeriv(const Eigen::Matrix3d& R0, const Eigen::Vector3d& r0,
                                                        const Eigen::Vector3d& bx0, const Eigen::Vector3d& p,
                                                        const Eigen::Matrix<double, 3, 6>& dr_g0,
                                                        const Eigen::Matrix<double, 3, 6>& dbx_g0) {
    return dbx_g0.transpose() * dr_g0 + dr_g0.transpose() * dbx_g0 +
           stkRot(R0, Eigen::Vector3d(r0 - p), Eigen::Vector3d::UnitX()) + stkPoint(R0, bx0, Eigen::Vector3d::Zero());
}

// SOC block (Cone/Frustum share it): dS_soc = -a_soc^T ( d(R^T r)/dxi
// - d(R^T p)/dxi ), r the pose origin, a_soc = E z_soc (E symmetric).
static Eigen::Matrix<double, 6, 6> coneSocSecondDeriv(const Eigen::Matrix3d& R0, const Eigen::Vector3d& p0,
                                                       const Eigen::Vector3d& a_soc, const Eigen::Vector3d& p) {
    return -(stkInvPoint(R0, p0, a_soc) - stkInvRot(R0, a_soc, p));
}

Eigen::Matrix<double, 6, 6> coneHessianFrozen(const Cone& shape, const Eigen::Matrix4d& g0, const Eigen::Vector3d& p,
                                               double z_ort, const Eigen::Vector3d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d p0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx0 = R0 * Eigen::Vector3d::UnitX();

    const Eigen::Vector3d Ediag(std::tan(shape.beta), 1.0, 1.0);
    const Eigen::Vector3d a_soc = Ediag.asDiagonal() * z_soc;

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    const Eigen::Matrix<double, 6, 6> dS_soc = coneSocSecondDeriv(R0, p0, a_soc, p);
    const Eigen::Matrix<double, 6, 6> dS_ort = z_ort * coneOrtQSecondDeriv(R0, p0, bx0, p, dr_g0, dbx_g0);

    return -(dS_soc + dS_ort);
}

// SOC term is Cone's verbatim; the ORT term is the same q-row second
// derivative as Cone's, weighted by (z_ort(0) - z_ort(1)) since the top cap's
// q is the base cap's negated (plus a constant alpha term that drops).
Eigen::Matrix<double, 6, 6> truncatedConeHessianFrozen(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                        const Eigen::Vector3d& p, const Eigen::Vector2d& z_ort,
                                                        const Eigen::Vector3d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d p0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d bx0 = R0 * Eigen::Vector3d::UnitX();

    const Eigen::Vector3d Ediag(shape.tan_beta, 1.0, 1.0);
    const Eigen::Vector3d a_soc = Ediag.asDiagonal() * z_soc;

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, Eigen::Vector3d::Zero());
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, Eigen::Vector3d::UnitX());

    const Eigen::Matrix<double, 6, 6> dS_soc = coneSocSecondDeriv(R0, p0, a_soc, p);
    const Eigen::Matrix<double, 6, 6> dS_ort =
        (z_ort(0) - z_ort(1)) * coneOrtQSecondDeriv(R0, p0, bx0, p, dr_g0, dbx_g0);

    return -(dS_soc + dS_ort);
}

Eigen::Matrix<double, 6, 6> ellipsoidHessianFrozen(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                    const Eigen::Vector3d& p, const Eigen::Vector4d& z_soc) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d p0 = g0.block<3, 1>(0, 3);
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    // grad SOC = a_ell^T ( d(R^T r)/dxi - d(R^T p)/dxi ), r the pose
    // origin, a_ell = U^T z_vec.
    const Eigen::Vector3d a_ell = shape.U.transpose() * z_vec;
    return stkInvPoint(R0, p0, a_ell) - stkInvRot(R0, a_ell, p);
}

} // namespace dcolpp::socp
