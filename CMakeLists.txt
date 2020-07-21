cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(PhysIKA LANGUAGES CXX CUDA)                                                             #指定本项目的编译语言为C++、CUDA

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性


set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式

set(CMAKE_CUDA_FLAGS "-arch=sm_30")                                         #默认为所有cuda代码生成computability3.0下二进制代码

include_directories("Source")                                                              #默认为本项目下各编译目标添加Source/ 目录

# temporary solution to solve linux compilation problem. our header file path should begin with Core this level
include_directories("Source/Core" "Source/Dynamics" "Source/Framework" "Source/GUI" "Source/IO" "Source/Rendering")

include_directories("Extern")                                           #默认为本项目下各编译目标添加Source/Physika_Dependency/ 目录
include_directories("Extern/OpenGL")                                    #默认为本项目下各编译目标添加Source/Physika_Dependency/OpenGL 目录
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})                                 	                #默认为本项目下各编译目标添加cuda/include 目录
file(COPY "Media/" DESTINATION "Media")
file(COPY "Source/Rendering/Shader/" DESTINATION "Shader")


option(PhysIKA_Python_Binding "Enable python binding with pybind11" ON)
if(PhysIKA_Python_Binding)
  add_subdirectory(Python)
endif()

option(PhysIKA_Tests "Enable building PhysIKA tests" ON)
if(PhysIKA_Tests)
  add_subdirectory(Extern/gtest)
  add_subdirectory(Tests)
endif()

option(PhysIKA_Qt5_GUI "Enable building Qt-based applications" OFF)

option(PhysIKA_Examples "Enable building examples" ON)
if(PhysIKA_Examples)
  add_subdirectory(Examples)
endif()

add_subdirectory(Source)



