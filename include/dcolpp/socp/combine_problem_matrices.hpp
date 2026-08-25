#pragma once
// dcolpp::socp — ported from DifferentiableCollisions.jl
// Source: src/combine_problem_matrices.jl (Kevin Tracy, MIT License). See
// NOTICE.md.
//
// Stacks two primitives' problem-matrix blocks into a single SOCP over a
// shared decision vector [p(3); alpha(1); extras1; extras2]. Most primitives
// contribute no extra decision variables (V=4); Capsule/Cylinder contribute
// one (axial parameter) and Polygon contributes two (local 2D coordinate).
// When both shapes have extras, each shape's own extra columns are padded
// with zeros in the other shape's rows, so neither shape's extra decision
// variable appears in the other's constraints. V1, V2 (and therefore which
// of the four layouts applies) are known at compile time here, so this is
// resolved with `if constexpr` instead of the Julia source's runtime
// branches on the same conditions.

#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

template <int N_ORT1, int N_SOC1, int N_ORT2, int N_SOC2, int NX>
struct CombinedProblem {
    static constexpr int n_ort = N_ORT1 + N_ORT2;
    static constexpr int n_soc1 = N_SOC1;
    static constexpr int n_soc2 = N_SOC2;
    static constexpr int nx = NX;
    static constexpr int ns = n_ort + n_soc1 + n_soc2;

    Vec<NX> c;
    Mat<ns, NX> G;
    Vec<ns> h;
};

template <int N_ORT1, int N_SOC1, int V1, int N_ORT2, int N_SOC2, int V2>
auto combineProblemMatrices(const ProblemMats<N_ORT1, N_SOC1, V1>& P1, const ProblemMats<N_ORT2, N_SOC2, V2>& P2) {
    constexpr int NS = N_ORT1 + N_ORT2 + N_SOC1 + N_SOC2;

    // h stacking is identical in every layout below.
    Vec<NS> h;
    h.template segment<N_ORT1>(0) = P1.h_ort;
    h.template segment<N_ORT2>(N_ORT1) = P2.h_ort;
    h.template segment<N_SOC1>(N_ORT1 + N_ORT2) = P1.h_soc;
    h.template segment<N_SOC2>(N_ORT1 + N_ORT2 + N_SOC1) = P2.h_soc;

    if constexpr (V1 == 4 && V2 == 4) {
        constexpr int NX = 4;
        Vec<NX> c = Vec<NX>::Zero(); c(3) = 1;
        Mat<NS, NX> G;
        G.template block<N_ORT1, 4>(0, 0) = P1.G_ort;
        G.template block<N_ORT2, 4>(N_ORT1, 0) = P2.G_ort;
        G.template block<N_SOC1, 4>(N_ORT1 + N_ORT2, 0) = P1.G_soc;
        G.template block<N_SOC2, 4>(N_ORT1 + N_ORT2 + N_SOC1, 0) = P2.G_soc;
        return CombinedProblem<N_ORT1, N_SOC1, N_ORT2, N_SOC2, NX>{c, G, h};

    } else if constexpr (V1 > 4 && V2 == 4) {
        constexpr int NX = V1;
        constexpr int extra = V1 - 4;
        Vec<NX> c = Vec<NX>::Zero(); c(3) = 1;
        Mat<NS, NX> G;
        G.template block<N_ORT1, V1>(0, 0) = P1.G_ort;
        G.template block<N_ORT2, 4>(N_ORT1, 0) = P2.G_ort;
        G.template block<N_ORT2, extra>(N_ORT1, 4).setZero();
        G.template block<N_SOC1, V1>(N_ORT1 + N_ORT2, 0) = P1.G_soc;
        G.template block<N_SOC2, 4>(N_ORT1 + N_ORT2 + N_SOC1, 0) = P2.G_soc;
        G.template block<N_SOC2, extra>(N_ORT1 + N_ORT2 + N_SOC1, 4).setZero();
        return CombinedProblem<N_ORT1, N_SOC1, N_ORT2, N_SOC2, NX>{c, G, h};

    } else if constexpr (V1 == 4 && V2 > 4) {
        constexpr int NX = V2;
        constexpr int extra = V2 - 4;
        Vec<NX> c = Vec<NX>::Zero(); c(3) = 1;
        Mat<NS, NX> G;
        G.template block<N_ORT1, 4>(0, 0) = P1.G_ort;
        G.template block<N_ORT1, extra>(0, 4).setZero();
        G.template block<N_ORT2, V2>(N_ORT1, 0) = P2.G_ort;
        G.template block<N_SOC1, 4>(N_ORT1 + N_ORT2, 0) = P1.G_soc;
        G.template block<N_SOC1, extra>(N_ORT1 + N_ORT2, 4).setZero();
        G.template block<N_SOC2, V2>(N_ORT1 + N_ORT2 + N_SOC1, 0) = P2.G_soc;
        return CombinedProblem<N_ORT1, N_SOC1, N_ORT2, N_SOC2, NX>{c, G, h};

    } else {
        static_assert(V1 > 4 && V2 > 4, "unreachable");
        constexpr int v1e = V1 - 4;
        constexpr int v2e = V2 - 4;
        constexpr int NX = V1 + v2e; // == V1 + V2 - 4
        Vec<NX> c = Vec<NX>::Zero(); c(3) = 1;
        Mat<NS, NX> G;

        G.template block<N_ORT1, V1>(0, 0) = P1.G_ort;
        G.template block<N_ORT1, v2e>(0, V1).setZero();

        G.template block<N_ORT2, 4>(N_ORT1, 0) = P2.G_ort.template block<N_ORT2, 4>(0, 0);
        G.template block<N_ORT2, v1e>(N_ORT1, 4).setZero();
        G.template block<N_ORT2, v2e>(N_ORT1, 4 + v1e) = P2.G_ort.template block<N_ORT2, v2e>(0, 4);

        G.template block<N_SOC1, V1>(N_ORT1 + N_ORT2, 0) = P1.G_soc;
        G.template block<N_SOC1, v2e>(N_ORT1 + N_ORT2, V1).setZero();

        G.template block<N_SOC2, 4>(N_ORT1 + N_ORT2 + N_SOC1, 0) = P2.G_soc.template block<N_SOC2, 4>(0, 0);
        G.template block<N_SOC2, v1e>(N_ORT1 + N_ORT2 + N_SOC1, 4).setZero();
        G.template block<N_SOC2, v2e>(N_ORT1 + N_ORT2 + N_SOC1, 4 + v1e) = P2.G_soc.template block<N_SOC2, v2e>(0, 4);

        return CombinedProblem<N_ORT1, N_SOC1, N_ORT2, N_SOC2, NX>{c, G, h};
    }
}

} // namespace dcolpp::socp
