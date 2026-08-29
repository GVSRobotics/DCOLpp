cmake_minimum_required(VERSION 3.16)
project(dcolpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DCOLPP_BUILD_TESTS "Build DCOL++ unit tests" ON)
option(DCOLPP_BUILD_EXAMPLES "Build DCOL++ example executables" ON)
option(DCOLPP_BUILD_BENCHMARKS "Build one-off benchmark executables" OFF)

# ---------------------------------------------------------------------------
# Eigen3: try a normal find_package first (system install, or vcpkg via
# CMAKE_PREFIX_PATH / the vcpkg toolchain file). Eigen is header-only, so if
# that fails we just fetch its source and point dcolpp_se3's include path at
# it directly -- deliberately NOT add_subdirectory-ing Eigen's own
# CMakeLists (which pulls in its C-language test/doc machinery we don't
# need).
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 NO_MODULE QUIET)
if(Eigen3_FOUND)
  add_library(dcolpp_se3 STATIC src/se3.cpp)
  target_link_libraries(dcolpp_se3 PUBLIC Eigen3::Eigen)
else()
  message(STATUS "Eigen3 not found via find_package; fetching headers via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
  )
  FetchContent_Populate(Eigen3)

  add_library(dcolpp_se3 STATIC src/se3.cpp)
  target_include_directories(dcolpp_se3 SYSTEM PUBLIC ${eigen3_SOURCE_DIR})
endif()

target_include_directories(dcolpp_se3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(dcolpp_se3 PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# dcolpp_socp : port of DifferentiableCollisions.jl. Mostly header-only,
# plus the plain-double analytic-derivative sources
# (src/socp_analytic_derivatives.cpp).
# ---------------------------------------------------------------------------
add_library(dcolpp_socp STATIC src/socp_analytic_derivatives.cpp)
target_link_libraries(dcolpp_socp PUBLIC dcolpp_se3)

# ---------------------------------------------------------------------------
# dcolpp_implicit : vendored iDCOL core + DCOL++'s added sensitivity (T)
# Added once src/implicit/*.cpp exist (Phase 4).
# ---------------------------------------------------------------------------
set(DCOLPP_IMPLICIT_SOURCES
  # populated in Phase 4
)
if(DCOLPP_IMPLICIT_SOURCES)
  add_library(dcolpp_implicit STATIC ${DCOLPP_IMPLICIT_SOURCES})
  target_include_directories(dcolpp_implicit PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(dcolpp_implicit PUBLIC dcolpp_se3)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DCOLPP_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(DCOLPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(DCOLPP_BUILD_BENCHMARKS)
  add_executable(bench_socp tools/bench_socp.cpp)
  target_link_libraries(bench_socp PRIVATE dcolpp_socp)
  target_compile_definitions(bench_socp PRIVATE NDEBUG EIGEN_NO_DEBUG)

  add_executable(bench_ergodic tools/bench_ergodic.cpp)
  target_link_libraries(bench_ergodic PRIVATE dcolpp_socp)
  target_compile_definitions(bench_ergodic PRIVATE NDEBUG EIGEN_NO_DEBUG)

  add_executable(bench_geo_init tools/bench_geo_init.cpp)
  target_link_libraries(bench_geo_init PRIVATE dcolpp_socp)
  target_compile_definitions(bench_geo_init PRIVATE NDEBUG EIGEN_NO_DEBUG)

  add_executable(bench_solve_vs_jacobian tools/bench_solve_vs_jacobian.cpp)
  target_link_libraries(bench_solve_vs_jacobian PRIVATE dcolpp_socp)
  target_compile_definitions(bench_solve_vs_jacobian PRIVATE NDEBUG EIGEN_NO_DEBUG)

  add_executable(bench_warm_start tools/bench_warm_start.cpp)
  target_link_libraries(bench_warm_start PRIVATE dcolpp_socp)
  target_compile_definitions(bench_warm_start PRIVATE NDEBUG EIGEN_NO_DEBUG)
endif()
