#pragma once
// A standard-normal generator with a fully self-contained, portable
// algorithm (Box-Muller over a hand-written [0,1) draw from mt19937's raw
// 32-bit output) -- NOT std::normal_distribution, whose exact algorithm is
// implementation-defined (only mt19937's bit sequence is standardized by
// the C++ standard). Confirmed directly: GCC/libstdc++ and
// LLVM-mingw's libc++ produce completely different double sequences from
// std::normal_distribution given the identical mt19937 seed, so a fixed
// (seed, trial index) test landed on a completely different random pose
// under each compiler -- occasionally landing one of them on the ~4%(-ish)
// -occurrence known ill-conditioned poses documented in DEVIATIONS.md §1b/
// §5 while the other compiler's draw for that same nominal trial happened
// not to. This makes every trial identical across any conforming
// implementation, so the test suite means the same thing under every
// compiler. Drop-in call syntax (operator()(rng)) matches
// std::normal_distribution<double> exactly, so swapping it in at call
// sites is a pure type-name replacement.
#include <cmath>
#include <cstdint>

namespace dcolpp_test {

class PortableNormal {
public:
    explicit PortableNormal(double mean = 0.0, double stddev = 1.0) : mean_(mean), stddev_(stddev) {}

    template <typename Rng>
    double operator()(Rng& rng) const {
        double u1 = uniform01(rng);
        if (u1 < 1e-300) u1 = 1e-300; // avoid log(0) at the (measure-zero) low end
        const double u2 = uniform01(rng);
        const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
        return mean_ + stddev_ * z;
    }

private:
    // 53 bits of mantissa from two of the generator's raw outputs -- a
    // fixed, hand-written formula (not std::uniform_real_distribution,
    // which has the same implementation-defined-algorithm issue).
    template <typename Rng>
    static double uniform01(Rng& rng) {
        const std::uint64_t a = static_cast<std::uint32_t>(rng()) >> 5; // 27 bits
        const std::uint64_t b = static_cast<std::uint32_t>(rng()) >> 6; // 26 bits
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0); // / 2^53
    }

    double mean_, stddev_;
};

} // namespace dcolpp_test
