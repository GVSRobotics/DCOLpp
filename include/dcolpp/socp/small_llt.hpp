#pragma once
// A compile-time-fixed-size Cholesky factor-and-solve, replacing
// Eigen::LLT for the tiny (3-7 row) SOC blocks and the nx-row Newton system
// in solver.hpp/nt_scaling.hpp. Same algorithm as LLT (no pivoting,
// factor once then forward/backward-substitute -- never an explicit
// inverse, per the numerical-stability note in nt_scaling.hpp), just
// without Eigen::LLT's machinery for general (possibly large, possibly
// rank-deficient) matrices, which is unrolled-loop overhead this library
// never needs: every matrix here is a fixed, small, PD-by-construction
// (or safely near-PD) SOC arrow/NT-scaling matrix.

#include <Eigen/Dense>
#include <cmath>
#include "dcolpp/socp/types.hpp"

namespace dcolpp::socp {

template <int N>
class SmallLLT {
public:
    SmallLLT() = default;
    explicit SmallLLT(const Eigen::Matrix<double, N, N>& A) { compute(A); }

    // Cholesky-Banachiewicz: L*L^T = A, A symmetric (only the lower
    // triangle is read).
    DCOLPP_INLINE void compute(const Eigen::Matrix<double, N, N>& A) {
        for (int j = 0; j < N; ++j) {
            double sum = A(j, j);
            for (int k = 0; k < j; ++k) sum -= L_(j, k) * L_(j, k);
            const double ljj = std::sqrt(sum);
            L_(j, j) = ljj;
            const double inv_ljj = 1.0 / ljj;
            invDiag_(j) = inv_ljj;
            for (int i = j + 1; i < N; ++i) {
                double s = A(i, j);
                for (int k = 0; k < j; ++k) s -= L_(i, k) * L_(j, k);
                L_(i, j) = s * inv_ljj;
            }
        }
    }

    // A \ b via forward + backward substitution. Multiplies by the
    // precomputed reciprocal diagonal (invDiag_) instead of dividing by
    // L_(i,i) -- same result, VTune (hotspots, user-mode sampling) showed
    // these per-row divisions (recomputed identically for every column in
    // the matrix overload below) as ~20% of total solve time on their own.
    // (Tried plain C-array storage for L_/invDiag_ instead of
    // Eigen::Matrix members -- measured slower, reverted.)
    DCOLPP_INLINE Eigen::Matrix<double, N, 1> solve(const Eigen::Matrix<double, N, 1>& b) const {
        Eigen::Matrix<double, N, 1> y;
        for (int i = 0; i < N; ++i) {
            double s = b(i);
            for (int k = 0; k < i; ++k) s -= L_(i, k) * y(k);
            y(i) = s * invDiag_(i);
        }
        Eigen::Matrix<double, N, 1> x;
        for (int i = N - 1; i >= 0; --i) {
            double s = y(i);
            for (int k = i + 1; k < N; ++k) s -= L_(k, i) * x(k);
            x(i) = s * invDiag_(i);
        }
        return x;
    }

    // A \ B, columnwise, via direct forward/back substitution indexed
    // straight into B -- Derived may be any fixed-N-row Eigen expression
    // (e.g. a Block from .middleRows<N>(...)), not just a concrete Matrix.
    // (Not a per-column call into the vector solve() above: that went
    // through an Eigen::Matrix<double,N,1> temporary per column: measured
    // to cost real time over nx>1 columns, unlike the same trick for a
    // single vector, where -O3 already inlined it away.)
    // Forward+backward substitution fused per column (not two passes over
    // the whole matrix): each column's intermediate y-values live in a
    // small local array, not a materialized N x NX matrix round-tripped
    // through memory between the two passes.
    template <typename Derived>
    DCOLPP_INLINE Eigen::Matrix<double, N, Derived::ColsAtCompileTime> solve(const Eigen::MatrixBase<Derived>& B) const {
        constexpr int NX = Derived::ColsAtCompileTime;
        Eigen::Matrix<double, N, NX> X;
        for (int c = 0; c < NX; ++c) {
            double y[N];
            for (int i = 0; i < N; ++i) {
                double s = B(i, c);
                for (int k = 0; k < i; ++k) s -= L_(i, k) * y[k];
                y[i] = s * invDiag_(i);
            }
            for (int i = N - 1; i >= 0; --i) {
                double s = y[i];
                for (int k = i + 1; k < N; ++k) s -= L_(k, i) * X(k, c);
                X(i, c) = s * invDiag_(i);
            }
        }
        return X;
    }

private:
    Eigen::Matrix<double, N, N> L_ = Eigen::Matrix<double, N, N>::Zero();
    Eigen::Matrix<double, N, 1> invDiag_ = Eigen::Matrix<double, N, 1>::Zero();
};

// A^T*A, lower triangle only (upper triangle left uninitialized) -- for
// feeding straight into SmallLLT::compute(), which never reads the upper
// triangle anyway. Half the multiply-adds of Eigen's A.transpose()*A (which
// computes the full symmetric result), and skips forming A.transpose()
// entirely: reads A column-major, matching its actual (column-major)
// storage, so each inner-k loop is a contiguous stride-1 read of A.
template <int K, int N>
DCOLPP_INLINE Eigen::Matrix<double, N, N> gramLower(const Eigen::Matrix<double, K, N>& A) {
    Eigen::Matrix<double, N, N> out;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A(k, i) * A(k, j);
            out(i, j) = s;
        }
    }
    return out;
}

// N=0 specialization (an empty ORT-only or single-SOC-block problem still
// instantiates SmallLLT<0> in generic code paths).
template <>
class SmallLLT<0> {
public:
    SmallLLT() = default;
    explicit SmallLLT(const Eigen::Matrix<double, 0, 0>&) {}
    void compute(const Eigen::Matrix<double, 0, 0>&) {}
    Eigen::Matrix<double, 0, 1> solve(const Eigen::Matrix<double, 0, 1>&) const { return {}; }
    template <int NX>
    Eigen::Matrix<double, 0, NX> solve(const Eigen::Matrix<double, 0, NX>&) const {
        return Eigen::Matrix<double, 0, NX>{};
    }
};

} // namespace dcolpp::socp
