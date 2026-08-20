#pragma once
// dcolpp::Dual6 — a small forward-mode dual-number scalar carrying a value
// and a fixed 6-dimensional vector of partial derivatives.
//
// This exists so DCOL++ can differentiate proximity queries with respect to
// a body's 6-dof local SE(3) twist without depending on a third-party
// autodiff library. It plays the same role here that `ForwardDiff.Dual`
// plays in the original DifferentiableCollisions.jl: template a small,
// closed-form piece of math (there: `kkt_R`/`problem_matrices`; here also
// iDCOL's `eval_F`) over the scalar type, seed the inputs with unit
// derivative directions, and read off the Jacobian from the outputs' `grad()`.
//
// Only the operators actually needed by that closed-form math are provided.

#include <Eigen/Core>
#include <cmath>

namespace dcolpp {

class Dual6 {
public:
    using Deriv = Eigen::Matrix<double, 6, 1>;

    Dual6() : val_(0.0), grad_(Deriv::Zero()) {}
    Dual6(double v) : val_(v), grad_(Deriv::Zero()) {}                 // NOLINT: implicit, needed for mixed double/Dual6 arithmetic
    Dual6(double v, const Deriv& g) : val_(v), grad_(g) {}

    // A Dual6 seeded as the `dir`-th (0..5) independent variable at value v.
    static Dual6 seed(double v, int dir) {
        Deriv g = Deriv::Zero();
        g(dir) = 1.0;
        return Dual6(v, g);
    }

    double value() const { return val_; }
    const Deriv& grad() const { return grad_; }

    Dual6 operator-() const { return Dual6(-val_, -grad_); }

    Dual6& operator+=(const Dual6& o) { val_ += o.val_; grad_ += o.grad_; return *this; }
    Dual6& operator-=(const Dual6& o) { val_ -= o.val_; grad_ -= o.grad_; return *this; }
    Dual6& operator*=(const Dual6& o) {
        grad_ = val_ * o.grad_ + o.val_ * grad_;
        val_ *= o.val_;
        return *this;
    }
    Dual6& operator/=(const Dual6& o) {
        const double inv = 1.0 / o.val_;
        grad_ = (grad_ * o.val_ - val_ * o.grad_) * (inv * inv);
        val_ *= inv;
        return *this;
    }

    friend Dual6 operator+(Dual6 a, const Dual6& b) { a += b; return a; }
    friend Dual6 operator-(Dual6 a, const Dual6& b) { a -= b; return a; }
    friend Dual6 operator*(Dual6 a, const Dual6& b) { a *= b; return a; }
    friend Dual6 operator/(Dual6 a, const Dual6& b) { a /= b; return a; }

    friend bool operator<(const Dual6& a, const Dual6& b) { return a.val_ < b.val_; }
    friend bool operator>(const Dual6& a, const Dual6& b) { return a.val_ > b.val_; }
    friend bool operator<=(const Dual6& a, const Dual6& b) { return a.val_ <= b.val_; }
    friend bool operator>=(const Dual6& a, const Dual6& b) { return a.val_ >= b.val_; }
    friend bool operator==(const Dual6& a, const Dual6& b) { return a.val_ == b.val_; }
    friend bool operator!=(const Dual6& a, const Dual6& b) { return a.val_ != b.val_; }

    // ADL hooks used by Eigen::numext (which does `using std::sqrt; sqrt(x);`)
    // and by any code that calls these unqualified on a Dual6.
    friend Dual6 sqrt(const Dual6& x) {
        const double s = std::sqrt(x.val_);
        // d/dx sqrt(x) = 1/(2 sqrt(x)); guard the s==0 case (not hit on any
        // path DCOL++ takes -- se3::retract never produces a zero here --
        // but fail safely rather than divide by zero).
        const double dsdx = (s > 0.0) ? (0.5 / s) : 0.0;
        return Dual6(s, dsdx * x.grad_);
    }
    friend Dual6 abs(const Dual6& x) {
        return (x.val_ < 0.0) ? -x : x;
    }

private:
    double val_;
    Deriv grad_;
};

} // namespace dcolpp

namespace Eigen {

template <>
struct NumTraits<dcolpp::Dual6> : NumTraits<double> {
    using Real = dcolpp::Dual6;
    using NonInteger = dcolpp::Dual6;
    using Nested = dcolpp::Dual6;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 6,
        AddCost = 6,
        MulCost = 12,
    };
};

} // namespace Eigen
