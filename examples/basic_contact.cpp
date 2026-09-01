// examples/basic_contact.cpp
//
// A cube vs. a cone: the contact query, its analytic Jacobian, and a short
// warm-started sweep. Build with -DDCOLPP_BUILD_EXAMPLES=ON, then run
// ./build/examples/basic_contact.

#include <Eigen/Dense>
#include <cstdio>

#include "dcolpp/se3.hpp"
#include "dcolpp/socp/contact.hpp"

using namespace dcolpp::socp;

int main() {
    // --- shapes, each in its own frame -----------------------------------
    Eigen::Matrix<double, 6, 3> A;
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    const Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    const Polytope<6> cube(A, b);          // body 1, fixed at the origin
    const Cone cone(1.5, 0.4363);          // body 2, height 1.5, ~25 deg half-angle

    // --- one query -----------------------------------------------------------
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g(2, 3) = 1.6;                          // cone shifted +1.6 along z

    SocpOptions opt;
    opt.compute_contact_manifold = true;

    const auto r = proximityContactJacobian(cube, cone, g, opt);
    std::printf("cube vs cone at z = %.2f\n", g(2, 3));
    std::printf("  converged     : %s (%d iters)\n", r.converged ? "yes" : "no", r.iters);
    std::printf("  alpha         : %.6f   (<1 hit, =1 touch, >1 apart)\n", r.alpha);
    std::printf("  witness       : %8.4f %8.4f %8.4f\n", r.witness_point.x(), r.witness_point.y(),
                r.witness_point.z());
    std::printf("  body1 witness : %8.4f %8.4f %8.4f\n", r.witness_body1.x(), r.witness_body1.y(),
                r.witness_body1.z());
    std::printf("  body2 witness : %8.4f %8.4f %8.4f\n", r.witness_body2.x(), r.witness_body2.y(),
                r.witness_body2.z());
    std::printf("  gap           : %+.6f\n", r.gap);
    std::printf("  normal        : %8.4f %8.4f %8.4f\n", r.normal.x(), r.normal.y(), r.normal.z());
    std::printf("  manifold dim  : %d   (0 point / 1 line / 2 face)\n", r.contact_manifold_dim);

    std::printf("  d(alpha)/dxi  : ");
    for (int k = 0; k < 6; ++k) std::printf("%+8.4f ", r.jacobian(3, k));
    std::printf("   [ d/dw (3) | d/dv (3) ]\n\n");

    // --- a short trajectory, warm-started --------------------------------
    // Small steps stay inside the warm handle's trust radius, so after the
    // first couple of solves each query reconverges in ~1 iteration.
    ContactWarmState<Polytope<6>, Cone> ws;
    std::printf("warm-started descent (cone lowered toward the cube):\n");
    for (double z = 1.25; z >= 1.10; z -= 0.015) {
        Eigen::Matrix4d gz = Eigen::Matrix4d::Identity();
        gz(2, 3) = z;
        const auto step = proximityContactJacobian(cube, cone, gz, opt, &ws);
        std::printf("  z = %.3f   alpha = %.6f   gap = %+.5f   iters = %d\n", z, step.alpha, step.gap, step.iters);
    }
    return 0;
}
