cmake_minimum_required(VERSION 3.10)

set(PHYSIKA_INCLUDE_PATH ${CMAKE_BINARY_DIR}/include CACHE STRING "include path in binary directory" FORCE)
set(PHYSIKA_LIB_CMAKES "")
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES "")

add_subdirectory(Core)
add_subdirectory(Framework)
add_subdirectory(IO)
add_subdirectory(Rendering)

add_subdirectory(Dynamics)
add_subdirectory(GUI)

# Write/install version file
include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/PhysIKAConfigVersion.cmake")
write_basic_package_version_file(${version_file}
    VERSION ${PHYSIKA_LIBRARY_VERSION}
    COMPATIBILITY AnyNewerVersion)

get_property(PHYSIKA_LIB_CMAKES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")

configure_file(PhysIKAConfig.cmake.in PhysIKAConfig.cmake @ONLY)
install(FILES   "${CMAKE_CURRENT_BINARY_DIR}/PhysIKAConfig.cmake"
                ${version_file}
        DESTINATION ${PHYSIKA_CMAKE_CONFIG_INSTALL_DIR}
        )