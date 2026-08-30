// Per-shape chain-rule derivatives and H_frozen. Declarations + overview
// in dcolpp/socp/analytic_derivatives.hpp.
#include "dcolpp/socp/analytic_derivatives.hpp"

namespace dcolpp::socp {

ShapeXiDerivative<0, 4, 4> sphereXiDerivative(const Sphere& shape, const Eigen::Matrix4d& g0) {
    // Stage 1+2 (generic, shape-independent): d(placed center r)/dxi.
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);

    // Stage 3 (shape-specific): h_soc = (0; -r), so d(h_soc)/dr = -I (the
    // rest of G_soc, h_soc(0) is constant, independent of r or Q). Chain
    // rule: d(h_soc)/dxi = d(h_soc)/dr * dr/dxi. Row 0 (h_soc's constant
    // entry) stays at ShapeXiDerivative's default zero.
    const Eigen::Matrix3d dHsoc_dr = -Eigen::Matrix3d::Identity();

    ShapeXiDerivative<0, 4, 4> out;
    out.dHsoc.block<3, 6>(1, 0) = dHsoc_dr * dr_dxi;
    return out;
}

ShapeXiDerivative<2, 4, 5> capsuleXiDerivative(const Capsule& shape, const Eigen::Matrix4d& g0, double t,
                                                const Eigen::Vector4d& z_soc) {
    // Stage 1+2 (generic): d(placed center r)/dxi, d(axis bx = Q*e1 =
    // R*(R_offset*e1))/dxi -- bx is a direction (homogeneous [bx;0] =
    // g*[R_offset*e1;0]), so it's dRotatedVectorDXi with the *local* axis
    // R_offset*e1 (not e1 itself -- R_offset is only Identity by default,
    // not in general), not dPointDXi.
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, shape.R_offset.col(0));

    // Stage 3 (shape-specific): x_local=(p,alpha,t), z_soc=(z0,z1,z2,z3) are
    // frozen (converged) values -- only Q and r vary with xi.
    ShapeXiDerivative<2, 4, 5> out;

    // G_soc*x_local = (-R_cap*alpha; -p + bx*t): only the bx*t term
    // (t frozen) carries xi-dependence, via bx alone. Chain rule:
    // d(G_soc*x_local)/dxi = t * d(bx)/dxi.
    out.dGsocX.block<3, 6>(1, 0) = t * dbx_dxi;

    // h_soc = (0; -r): d(h_soc)/dr = -I, chain rule as in sphereXiDerivative.
    out.dHsoc.block<3, 6>(1, 0) = -dr_dxi;

    // (G_soc^T*z_soc)_4 = bx . z_soc's spatial part -- the only entry that
    // involves Q at all (the rest pick out -z_soc's components or multiply
    // by the constant radius, both xi-independent). Chain rule:
    // d(bx . z_vec)/dxi = z_vec . d(bx)/dxi, z_vec frozen.
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    out.dGsocTz.row(4) = z_vec.transpose() * dbx_dxi;

    // G_ort, h_ort are both constant (no Q or r dependence) -- their
    // derivatives, and dGortTz, stay at the struct's default zero.
    return out;
}

ShapeXiDerivative<4, 4, 5> cylinderXiDerivative(const Cylinder& shape, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                 const Eigen::Vector4d& z_ort) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d r = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d bx = R0 * shape.R_offset.col(0);
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, shape.R_offset.col(0));

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
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    const Eigen::Vector3d bx0 = R0 * axis_local;

    const double tanb = std::tan(shape.beta);
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, axis_local);

    ShapeXiDerivative<1, 3, 4> out;

    // G_soc*x_local = -E*Q^T*p + (const)*alpha: only E*Q^T*p carries
    // xi-dependence, via Q^T*p = R_offset^T*(R^T*p) -- p is a frozen
    // decision variable (no product rule needed, unlike h_soc below).
    out.dGsocX = -E * shape.R_offset.transpose() * se3::dInverseRotatedVectorDXi(g0, p);

    // h_soc = -E*Q^T*r: r = r(xi) also varies with xi, so Q^T*r needs the
    // ordinary product rule -- one term treats R^T as frozen at R0^T
    // (giving R0^T*dr_dxi), the other treats r as frozen at its current
    // value r0 (giving dInverseRotatedVectorDXi evaluated there, not at a
    // decision variable).
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc = -E * shape.R_offset.transpose() * dRtr_dxi;

    // (G_soc^T*z_soc)'s first 3 entries = -Q*E*z_soc = -R*(R_offset*E*z_soc)
    // -- R_offset, E, z_soc are all xi-independent, so this is a single
    // dRotatedVectorDXi call on that fixed local vector.
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, shape.R_offset * E * z_soc);

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
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    const Eigen::Vector3d bx0 = R0 * axis_local;

    const double tanb = shape.tan_beta;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_dxi = se3::dRotatedVectorDXi(g0, axis_local);

    ShapeXiDerivative<2, 3, 4> out;

    // --- SOC block: identical to Cone (see coneXiDerivative for the notes) ---
    out.dGsocX = -E * shape.R_offset.transpose() * se3::dInverseRotatedVectorDXi(g0, p);
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc = -E * shape.R_offset.transpose() * dRtr_dxi;
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, shape.R_offset * E * z_soc);

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
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Matrix<double, 3, 6> dr_dxi = se3::dPointDXi(g0, shape.r_offset);

    ShapeXiDerivative<0, 4, 4> out;

    // G_soc*x_local = (-alpha; -U*Q^T*p): only U*Q^T*p carries
    // xi-dependence, via Q^T*p -- p frozen, no product rule (same pattern
    // as Cone's dGsocX, U standing in for E).
    out.dGsocX.block<3, 6>(1, 0) = -shape.U * shape.R_offset.transpose() * se3::dInverseRotatedVectorDXi(g0, p);

    // h_soc = (0; -U*Q^T*r): r = r(xi) also varies -- product rule, same
    // pattern as Cone's h_soc.
    const Eigen::Matrix<double, 3, 6> dRtr_dxi = se3::dInverseRotatedVectorDXi(g0, r0) + R0.transpose() * dr_dxi;
    out.dHsoc.block<3, 6>(1, 0) = -shape.U * shape.R_offset.transpose() * dRtr_dxi;

    // (G_soc^T*z_soc)'s first 3 entries = -Q*U^T*z_vec =
    // -R*(R_offset*U^T*z_vec) -- R_offset, U, z_vec all xi-independent.
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    out.dGsocTz.block<3, 6>(0, 0) = -se3::dRotatedVectorDXi(g0, shape.R_offset * shape.U.transpose() * z_vec);

    return out;
}

// H_frozen (directional): d/dt[grad_shape_contribution(g0*Exp(t*d), ...)]|_0.
// Each is the same Stage-3 q-formula as the corresponding *XiDerivative
// above, differentiated once more w.r.t. the outer perturbation: se3::d2*DXi
// in place of se3::d*DXi, full product rule wherever the first derivative
// already needed one.

Eigen::Matrix<double, 1, 6> sphereHessianFrozen(const Sphere& shape, const Eigen::Matrix4d& g0,
                                                 const Eigen::Vector4d& z_soc, const se3::Vector6d& d) {
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    return z_vec.transpose() * se3::d2PointDXi(g0, shape.r_offset, d);
}

Eigen::Matrix<double, 1, 6> capsuleHessianFrozen(const Capsule& shape, const Eigen::Matrix4d& g0, double t,
                                                  const Eigen::Vector4d& z_soc, const se3::Vector6d& d) {
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    return z_vec.transpose() *
           (se3::d2PointDXi(g0, shape.r_offset, d) + t * se3::d2RotatedVectorDXi(g0, axis_local, d));
}

Eigen::Matrix<double, 1, 6> cylinderHessianFrozen(const Cylinder& shape, const Eigen::Matrix4d& g0,
                                                   const Eigen::Vector3d& p, double t, const Eigen::Vector4d& z_soc,
                                                   const Eigen::Vector4d& z_ort, const se3::Vector6d& d) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d bx0 = R0 * axis_local;

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, axis_local);
    const Eigen::Vector3d dr_dt = dr_g0 * d;
    const Eigen::Vector3d dbx_dt = dbx_g0 * d;
    const Eigen::Matrix<double, 3, 6> d2r = se3::d2PointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2bx = se3::d2RotatedVectorDXi(g0, axis_local, d);

    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    const Eigen::Matrix<double, 1, 6> dS_soc = -z_vec.transpose() * (d2r + t * d2bx);

    const Eigen::Matrix<double, 1, 6> d_bxdotr_dot =
        dr_dt.transpose() * dbx_g0 + r0.transpose() * d2bx + dbx_dt.transpose() * dr_g0 + bx0.transpose() * d2r;
    const Eigen::Matrix<double, 1, 6> dq_row2 = -d_bxdotr_dot + p.transpose() * d2bx;
    const Eigen::Matrix<double, 1, 6> dq_row3 = d_bxdotr_dot - p.transpose() * d2bx;
    const Eigen::Matrix<double, 1, 6> dS_ort = z_ort(2) * dq_row2 + z_ort(3) * dq_row3;

    return -(dS_soc + dS_ort);
}

Eigen::Matrix<double, 1, 6> coneHessianFrozen(const Cone& shape, const Eigen::Matrix4d& g0, const Eigen::Vector3d& p,
                                               double z_ort, const Eigen::Vector3d& z_soc, const se3::Vector6d& d) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d bx0 = R0 * axis_local;

    const double tanb = std::tan(shape.beta);
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, axis_local);
    const Eigen::Vector3d dr_dt = dr_g0 * d;
    const Eigen::Vector3d dbx_dt = dbx_g0 * d;
    const Eigen::Matrix<double, 3, 6> d2r = se3::d2PointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2bx = se3::d2RotatedVectorDXi(g0, axis_local, d);

    const Eigen::Matrix<double, 3, 6> d2Rtr = se3::d2InverseRotatedPointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2IRVp = se3::d2InverseRotatedVectorDXi(g0, p, d);
    const Eigen::Matrix<double, 1, 6> dS_soc = -z_soc.transpose() * (E * shape.R_offset.transpose() * (d2Rtr - d2IRVp));

    const Eigen::Matrix<double, 1, 6> dq_ort = dr_dt.transpose() * dbx_g0 + (r0 - p).transpose() * d2bx +
                                                dbx_dt.transpose() * dr_g0 + bx0.transpose() * d2r;
    const Eigen::Matrix<double, 1, 6> dS_ort = z_ort * dq_ort;

    return -(dS_soc + dS_ort);
}

// SOC term is coneHessianFrozen's verbatim; the ORT term is the same q-row
// second derivative as Cone's, but weighted by (z_ort(0) - z_ort(1)) since
// row 1's q is row 0's negated (plus a constant alpha term that drops).
Eigen::Matrix<double, 1, 6> truncatedConeHessianFrozen(const TruncatedCone& shape, const Eigen::Matrix4d& g0,
                                                        const Eigen::Vector3d& p, const Eigen::Vector2d& z_ort,
                                                        const Eigen::Vector3d& z_soc, const se3::Vector6d& d) {
    const Eigen::Matrix3d R0 = g0.block<3, 3>(0, 0);
    const Eigen::Vector3d axis_local = shape.R_offset.col(0);
    const Eigen::Vector3d r0 = g0.block<3, 1>(0, 3) + R0 * shape.r_offset;
    const Eigen::Vector3d bx0 = R0 * axis_local;

    const double tanb = shape.tan_beta;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    E(0, 0) = tanb;
    E(1, 1) = 1.0;
    E(2, 2) = 1.0;

    const Eigen::Matrix<double, 3, 6> dr_g0 = se3::dPointDXi(g0, shape.r_offset);
    const Eigen::Matrix<double, 3, 6> dbx_g0 = se3::dRotatedVectorDXi(g0, axis_local);
    const Eigen::Vector3d dr_dt = dr_g0 * d;
    const Eigen::Vector3d dbx_dt = dbx_g0 * d;
    const Eigen::Matrix<double, 3, 6> d2r = se3::d2PointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2bx = se3::d2RotatedVectorDXi(g0, axis_local, d);

    const Eigen::Matrix<double, 3, 6> d2Rtr = se3::d2InverseRotatedPointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2IRVp = se3::d2InverseRotatedVectorDXi(g0, p, d);
    const Eigen::Matrix<double, 1, 6> dS_soc = -z_soc.transpose() * (E * shape.R_offset.transpose() * (d2Rtr - d2IRVp));

    // q0 = bx.(r - p) + (L/2)*alpha ; q1 = -bx.(r - p) + (L/2)*alpha.
    const Eigen::Matrix<double, 1, 6> dq0 = dr_dt.transpose() * dbx_g0 + (r0 - p).transpose() * d2bx +
                                             dbx_dt.transpose() * dr_g0 + bx0.transpose() * d2r;
    const Eigen::Matrix<double, 1, 6> dS_ort = (z_ort(0) - z_ort(1)) * dq0;

    return -(dS_soc + dS_ort);
}

Eigen::Matrix<double, 1, 6> ellipsoidHessianFrozen(const Ellipsoid& shape, const Eigen::Matrix4d& g0,
                                                    const Eigen::Vector3d& p, const Eigen::Vector4d& z_soc,
                                                    const se3::Vector6d& d) {
    const Eigen::Matrix<double, 3, 6> d2Rtr = se3::d2InverseRotatedPointDXi(g0, shape.r_offset, d);
    const Eigen::Matrix<double, 3, 6> d2IRVp = se3::d2InverseRotatedVectorDXi(g0, p, d);
    const Eigen::Vector3d z_vec = z_soc.tail<3>();
    const Eigen::Matrix<double, 1, 6> dS =
        -z_vec.transpose() * (shape.U * shape.R_offset.transpose() * (d2Rtr - d2IRVp));
    return -dS;
}

} // namespace dcolpp::socp
