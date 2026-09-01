# Runnable examples of the public API. Built when DCOLPP_BUILD_EXAMPLES=ON.
add_executable(basic_contact basic_contact.cpp)
target_link_libraries(basic_contact PRIVATE dcolpp_socp)
target_compile_features(basic_contact PRIVATE cxx_std_17)
