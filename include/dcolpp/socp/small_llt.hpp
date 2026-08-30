#pragma once
// Fixed-size Cholesky factor-and-solve for tiny PD SOC and Newton-system
// matrices; same as LLT but without the general-purpose overhead.

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

    // A \ b via forward + backward substitution, multiplying by the
    // precomputed reciprocal diagonal invDiag_ rather than dividing by
    // L_(i,i) at each row.
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

    // A \ B, columnwise. Forward and backward substitution are fused per
    // column, indexed straight into B, with the intermediate y-values in a
    // small local array -- no per-column temporary, no N x NX round-trip
    // through memory. Derived may be any fixed-N-row Eigen expression (e.g.
    // a Block from .middleRows<N>(...)), not just a concrete Matrix.
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

// A^T*A, lower triangle only (upper left uninitialized) -- SmallLLT::compute()
// never reads the upper triangle. Skips forming A.transpose() and reads A
// column-major, so each inner-k loop is a stride-1 read.
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
