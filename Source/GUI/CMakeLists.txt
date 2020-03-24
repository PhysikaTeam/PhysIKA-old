cmake_minimum_required(VERSION 3.10)

set(LIB_NAMES GlutGUI QtGUI)

add_subdirectory(GlutGUI)

if(PhysIKA_Qt5_GUI)
    add_subdirectory(QtGUI)
endif()
