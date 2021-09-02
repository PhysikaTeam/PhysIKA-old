cmake_minimum_required(VERSION 3.10)
project(mass_spring)

add_library(ms_hg mass_spring_obj.cpp para.cpp)
install_as_LIB(ms_hg)
target_include_directories(ms_hg PUBLIC
$<BUILD_INTERFACE:${ZJU_ROOT_DIR}>
)