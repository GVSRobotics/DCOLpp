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

namespace dcolpp::socp {

template <int N>
class SmallLLT {
public:
    SmallLLT() = default;
    explicit SmallLLT(const Eigen::Matrix<double, N, N>& A) { compute(A); }

    // Cholesky-Banachiewicz: L*L^T = A, A symmetric (only the lower
    // triangle is read).
    void compute(const Eigen::Matrix<double, N, N>& A) {
        for (int j = 0; j < N; ++j) {
            double sum = A(j, j);
            for (int k = 0; k < j; ++k) sum -= L_(j, k) * L_(j, k);
            const double ljj = std::sqrt(sum);
            L_(j, j) = ljj;
            const double inv_ljj = 1.0 / ljj;
            for (int i = j + 1; i < N; ++i) {
                double s = A(i, j);
                for (int k = 0; k < j; ++k) s -= L_(i, k) * L_(j, k);
                L_(i, j) = s * inv_ljj;
            }
        }
    }

    // A \ b via forward + backward substitution.
    Eigen::Matrix<double, N, 1> solve(const Eigen::Matrix<double, N, 1>& b) const {
        Eigen::Matrix<double, N, 1> y;
        for (int i = 0; i < N; ++i) {
            double s = b(i);
            for (int k = 0; k < i; ++k) s -= L_(i, k) * y(k);
            y(i) = s / L_(i, i);
        }
        Eigen::Matrix<double, N, 1> x;
        for (int i = N - 1; i >= 0; --i) {
            double s = y(i);
            for (int k = i + 1; k < N; ++k) s -= L_(k, i) * x(k);
            x(i) = s / L_(i, i);
        }
        return x;
    }

    // A \ B, columnwise. Derived may be any fixed-N-row Eigen expression
    // (e.g. a Block from .middleRows<N>(...)), not just a concrete Matrix.
    template <typename Derived>
    Eigen::Matrix<double, N, Derived::ColsAtCompileTime> solve(const Eigen::MatrixBase<Derived>& B) const {
        constexpr int NX = Derived::ColsAtCompileTime;
        Eigen::Matrix<double, N, NX> X;
        for (int c = 0; c < NX; ++c) X.col(c) = solve(Eigen::Matrix<double, N, 1>(B.col(c)));
        return X;
    }

private:
    Eigen::Matrix<double, N, N> L_ = Eigen::Matrix<double, N, N>::Zero();
};

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
