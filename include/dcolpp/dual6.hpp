#pragma once
// dcolpp::DualN<T> — a small forward-mode dual-number scalar carrying a
// value and a fixed 6-dimensional vector of partial derivatives.
//
// `Dual6 = DualN<double>` differentiates a small, closed-form piece of math
// (`kktR`/`problemMatrices`) with respect to a body's 6-dof local SE(3)
// twist: template the function over the scalar type, seed the inputs with
// unit derivative directions, and read off the Jacobian from the outputs'
// `grad()`. `DualN<T>` stays templated on `T` only so `Dual6` is a plain
// generalization of the type it always was -- it is never nested in this
// library.
//
// Only the operators actually needed by the closed-form math in this
// library are provided.

#include <Eigen/Core>
#include <cmath>
#include <type_traits>

namespace dcolpp {

template <typename T>
class DualN {
public:
    using Deriv = Eigen::Matrix<T, 6, 1>;

    DualN() : val_(T(0.0)), grad_(Deriv::Zero()) {}
    DualN(T v) : val_(v), grad_(Deriv::Zero()) {}        // NOLINT: implicit, needed for mixed T/DualN<T> arithmetic
    DualN(T v, const Deriv& g) : val_(v), grad_(g) {}

    // Promote a plain double into DualN<T> with zero derivative, going
    // through T's own double-conversion (so DualN<Dual6>(0.5) works the same
    // way Dual6(0.5) already does). Only enabled for T != double: for
    // T = double this constructor would be the exact same signature as
    // `DualN(T v)` above and the class wouldn't compile, so SFINAE disables
    // it in exactly that case -- `DualN(T v)` alone already covers T=double.
    template <typename U = T, typename = std::enable_if_t<!std::is_same_v<U, double>>>
    DualN(double v) : val_(T(v)), grad_(Deriv::Zero()) {}  // NOLINT: implicit

    // A DualN seeded as the `dir`-th (0..5) independent variable at value v.
    static DualN seed(double v, int dir) {
        Deriv g = Deriv::Zero();
        g(dir) = T(1.0);
        return DualN(T(v), g);
    }

    const T& value() const { return val_; }
    const Deriv& grad() const { return grad_; }

    DualN operator-() const { return DualN(-val_, -grad_); }

    DualN& operator+=(const DualN& o) { val_ += o.val_; grad_ += o.grad_; return *this; }
    DualN& operator-=(const DualN& o) { val_ -= o.val_; grad_ -= o.grad_; return *this; }
    DualN& operator*=(const DualN& o) {
        grad_ = val_ * o.grad_ + o.val_ * grad_;
        val_ *= o.val_;
        return *this;
    }
    DualN& operator/=(const DualN& o) {
        const T inv = T(1.0) / o.val_;
        grad_ = (grad_ * o.val_ - val_ * o.grad_) * (inv * inv);
        val_ *= inv;
        return *this;
    }

    friend DualN operator+(DualN a, const DualN& b) { a += b; return a; }
    friend DualN operator-(DualN a, const DualN& b) { a -= b; return a; }
    friend DualN operator*(DualN a, const DualN& b) { a *= b; return a; }
    friend DualN operator/(DualN a, const DualN& b) { a /= b; return a; }

    friend bool operator<(const DualN& a, const DualN& b) { return a.val_ < b.val_; }
    friend bool operator>(const DualN& a, const DualN& b) { return a.val_ > b.val_; }
    friend bool operator<=(const DualN& a, const DualN& b) { return a.val_ <= b.val_; }
    friend bool operator>=(const DualN& a, const DualN& b) { return a.val_ >= b.val_; }
    friend bool operator==(const DualN& a, const DualN& b) { return a.val_ == b.val_; }
    friend bool operator!=(const DualN& a, const DualN& b) { return a.val_ != b.val_; }

    // ADL hooks used by Eigen::numext (which does `using std::sqrt; sqrt(x);`)
    // and by any code that calls these unqualified on a DualN. For T=Dual6,
    // `sqrt(x.val_)`/comparisons below resolve via ADL to Dual6's own hooks
    // (found alongside `std::sqrt` through the `using` declaration at the
    // call site in Eigen::numext), so nesting composes automatically.
    friend DualN sqrt(const DualN& x) {
        using std::sqrt;
        const T s = sqrt(x.val_);
        // d/dx sqrt(x) = 1/(2 sqrt(x)); guard the s==0 case (not hit on any
        // path DCOL++ takes -- se3::retract never produces a zero here --
        // but fail safely rather than divide by zero).
        const T dsdx = (s > T(0.0)) ? (T(0.5) / s) : T(0.0);
        return DualN(s, dsdx * x.grad_);
    }
    friend DualN abs(const DualN& x) {
        return (x.val_ < T(0.0)) ? -x : x;
    }
    friend DualN sin(const DualN& x) {
        using std::cos;
        using std::sin;
        return DualN(sin(x.val_), cos(x.val_) * x.grad_);
    }
    friend DualN cos(const DualN& x) {
        using std::cos;
        using std::sin;
        return DualN(cos(x.val_), -sin(x.val_) * x.grad_);
    }

private:
    T val_;
    Deriv grad_;
};

using Dual6 = DualN<double>;

} // namespace dcolpp

namespace Eigen {

// Note this inherits NumTraits<T>'s precision helpers (epsilon(),
// dummy_precision(), ...) UNCHANGED rather than overriding them to return a
// DualN<T> -- they stay double-valued even for Dual6. That simplification is
// safe here because this library only ever exercises basic arithmetic and
// Eigen::PartialPivLU/LLT through this type, neither of which consults
// those precision helpers.
template <typename T>
struct NumTraits<dcolpp::DualN<T>> : NumTraits<T> {
    using Real = dcolpp::DualN<T>;
    using NonInteger = dcolpp::DualN<T>;
    using Nested = dcolpp::DualN<T>;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 6 * NumTraits<T>::ReadCost,
        AddCost = 6 * NumTraits<T>::AddCost,
        MulCost = 12 * NumTraits<T>::MulCost,
    };
};

} // namespace Eigen
