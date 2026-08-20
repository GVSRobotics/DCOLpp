add_executable(dcolpp_tests
  test_scaffold.cpp
  test_se3_dual6.cpp
  test_socp_forward.cpp
  test_socp_julia_parity.cpp
)
target_include_directories(dcolpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dcolpp_tests PRIVATE dcolpp_socp Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(dcolpp_tests)
