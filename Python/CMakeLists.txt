cmake_minimum_required(VERSION 3.10)

add_subdirectory(${PROJECT_SOURCE_DIR}/Extern/pybind11 pybind11)

set(PyBind_Name PyPhysIKA)

link_libraries(Core Framework IO Rendering)
link_libraries(GlutGUI)

pybind11_add_module(${PyBind_Name} PyPhysIKA.cpp)
