set (build_libs dashenfu_lib)

add_executable(dashenfu dashenfu.cpp)
target_link_libraries(dashenfu ${build_libs})
