cmake_minimum_required(VERSION 3.16)
project(dcolpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DCOLPP_BUILD_TESTS "Build DCOL++ unit tests" ON)
option(DCOLPP_BUILD_EXAMPLES "Build DCOL++ example executables" ON)

# ---------------------------------------------------------------------------
# Eigen3: try a normal find_package first (system install, or vcpkg via
# CMAKE_PREFIX_PATH / the vcpkg toolchain file). Eigen is header-only, so if
# that fails we just fetch its source and point an INTERFACE target at it
# directly -- deliberately NOT add_subdirectory-ing Eigen's own CMakeLists
# (which pulls in its C-language test/doc machinery we don't need).
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 NO_MODULE QUIET)
if(Eigen3_FOUND)
  add_library(dcolpp_se3 INTERFACE)
  target_link_libraries(dcolpp_se3 INTERFACE Eigen3::Eigen)
else()
  message(STATUS "Eigen3 not found via find_package; fetching headers via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
  )
  FetchContent_Populate(Eigen3)

  add_library(dcolpp_se3 INTERFACE)
  target_include_directories(dcolpp_se3 SYSTEM INTERFACE ${eigen3_SOURCE_DIR})
endif()

target_include_directories(dcolpp_se3 INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(dcolpp_se3 INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# dcolpp_socp : header-only port of DifferentiableCollisions.jl
# ---------------------------------------------------------------------------
add_library(dcolpp_socp INTERFACE)
target_link_libraries(dcolpp_socp INTERFACE dcolpp_se3)

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
