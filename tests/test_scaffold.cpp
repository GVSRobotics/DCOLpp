#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "dcolpp/version.hpp"

TEST_CASE("scaffold: version constants are sane", "[scaffold]") {
    REQUIRE(dcolpp::kVersionMajor == 0);
}

TEST_CASE("scaffold: Eigen is usable", "[scaffold]") {
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    REQUIRE(g.trace() == 4.0);
}
