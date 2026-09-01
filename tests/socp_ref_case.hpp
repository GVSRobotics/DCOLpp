#pragma once
// Plain-data struct populated by tests/generated/socp_reference_data.hpp
// (auto-generated from the original Julia
// library). Kept separate from the generator's output so regenerating the
// data never has to also regenerate this definition.

namespace dcolpp::test_ref {

struct SocpRefCase {
    double g[16];       // relative pose g = g1^{-1} g2, row-major
    double alpha;        // expected proximity alpha
    double witness[3];   // expected witness point, in body-1 frame
};

} // namespace dcolpp::test_ref
