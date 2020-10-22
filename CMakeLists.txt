cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(PhysIKA LANGUAGES CXX CUDA)                                                             #指定本项目的编译语言为C++、CUDA

# define library version (update: apparently you can also do it in project()!)
set(PHYSIKA_LIBRARY_VERSION_MAJOR 2 CACHE STRING "PhysIKA major version" FORCE)
set(PHYSIKA_LIBRARY_VERSION_MINOR 2 CACHE STRING "PhysIKA minor version" FORCE)
set(PHYSIKA_LIBRARY_VERSION_PATCH 1 CACHE STRING "patch version of PhysIKA" FORCE)
set(PHYSIKA_LIBRARY_VERSION "${PHYSIKA_LIBRARY_VERSION_MAJOR}.${PHYSIKA_LIBRARY_VERSION_MINOR}.${PHYSIKA_LIBRARY_VERSION_PATCH}" CACHE STRING "PhysIKA version" FORCE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性


set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式

set(CMAKE_CUDA_FLAGS "-arch=sm_60")                                         #默认为所有cuda代码生成computability7.0下二进制代码

file(COPY "Media/" DESTINATION "Media")
file(COPY "Source/Rendering/Shader/" DESTINATION "Shader")

#set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

IF(MSVC)
    SET( CMAKE_DEBUG_POSTFIX "d" )
ENDIF(MSVC)

# installation destinations
if(UNIX)
    include(GNUInstallDirs)

    set(PHYSIKA_INC_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/PhysIKA")
    set(PHYSIKA_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
    set(PHYSIKA_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(PHYSIKA_ARCHIVE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(PHYSIKA_FRAMEWORK_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")

    set(PHYSIKA_CMAKE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/PhysIKA/cmake")
    set(PHYSIKA_ADDITIONAL_FILES_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/PhysIKA")

    set(PHYSIKA_RUNTIME_INSTALL_DIR "bin") # for the nodesize_dbg, just ignore version and the like
    set(PHYSIKA_INC_INSTALL_DIR "include/PhysIKA") # header filestree

elseif(WIN32)
    set(PHYSIKA_INC_INSTALL_DIR "include/PhysIKA")
    set(PHYSIKA_RUNTIME_INSTALL_DIR   "bin")
    set(PHYSIKA_LIBRARY_INSTALL_DIR   "bin")
    set(PHYSIKA_ARCHIVE_INSTALL_DIR   "lib")
    set(PHYSIKA_FRAMEWORK_INSTALL_DIR "bin")

    set(PHYSIKA_CMAKE_CONFIG_INSTALL_DIR "share/PhysIKA/cmake")
    set(PHYSIKA_ADDITIONAL_FILES_INSTALL_DIR "share/PhysIKA")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    message(FATAL_ERROR "Could not set install folders for this platform!")
endif()

option(PhysIKA_Python_Binding "Enable python binding with pybind11" ON)
if(PhysIKA_Python_Binding)
    add_subdirectory(Python)
endif()

option(PhysIKA_Tests "Enable building PhysIKA tests" ON)
if(PhysIKA_Tests)
    add_subdirectory(Extern/gtest)
    add_subdirectory(Tests)
endif()

add_subdirectory(Extern/glew-1.10.0)
add_subdirectory(Extern/freeglut-3.2.1)

option(PhysIKA_Qt5_GUI "Enable building Qt-based applications" OFF)

option(PhysIKA_Examples "Enable building examples" ON)
if(PhysIKA_Examples)
    add_subdirectory(Examples)
endif()

add_subdirectory(Source)


