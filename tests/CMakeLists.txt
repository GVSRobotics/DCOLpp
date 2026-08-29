add_executable(dcolpp_tests
  test_scaffold.cpp
  test_se3.cpp
  test_socp_forward.cpp
  test_socp_julia_parity.cpp
  test_socp_diff.cpp
  test_analytic_derivatives.cpp
  test_hessian_derivatives.cpp
  test_proximity_contact.cpp
  test_contact_degeneracy.cpp
  test_contact_manifold.cpp
  test_warm_start.cpp
)
target_include_directories(dcolpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dcolpp_tests PRIVATE dcolpp_socp Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(dcolpp_tests)
