add_executable(dcolpp_tests
  test_scaffold.cpp
  test_se3_dual6.cpp
)
target_link_libraries(dcolpp_tests PRIVATE dcolpp_se3 Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(dcolpp_tests)
