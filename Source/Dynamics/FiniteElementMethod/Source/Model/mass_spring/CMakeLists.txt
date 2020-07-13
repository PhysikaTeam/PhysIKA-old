cmake_minimum_required(VERSION 3.10)
project(mass_spring)

enable_language(Fortran)

add_library(edge_hg edge_gradient_and_hessian.f90)
add_library(ms_hg mass_spring_obj.cpp para.cpp)
TARGET_LINK_LIBRARIES(ms_hg edge_hg)

