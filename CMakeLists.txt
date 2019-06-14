cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(Physika LANGUAGES CXX CUDA)                                                             #指定本项目的编译语言为C++、CUDA
find_package(Qt5 COMPONENTS Core Widgets)                                                       #查找qt包,可选项

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性


set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式

set(CMAKE_CUDA_FLAGS "-arch=sm_30")                                         #默认为所有cuda代码生成computability3.0下二进制代码

include_directories("Engine")                                                              #默认为本项目下各编译目标添加Engine/ 目录

# temporary solution to solve linux compilation problem. our header file path should begin with Core this level
include_directories("Engine/Core" "Engine/Dynamics" "Engine/Framework" "Engine/GUI" "Engine/IO" "Engine/Rendering")


include_directories("Extern")                                           #默认为本项目下各编译目标添加Engine/Physika_Dependency/ 目录
include_directories("Extern/OpenGL")                                    #默认为本项目下各编译目标添加Engine/Physika_Dependency/OpenGL 目录
include_directories("Extern/irrLicht/include")                          #默认为本项目下各编译目标添加Engine/Physika_dependency/irrLicht/include 目录
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})                                 	                #默认为本项目下各编译目标添加cuda/include 目录


if(WIN32)                                                               
    link_directories("Extern/OpenGL/lib/Windows/X64")
    link_libraries(freeglut glew32 glu32 opengl32)
    link_directories("Extern/irrLicht/lib/Win64-visualStudio")              
    link_libraries(Irrlicht)
endif()
if(APPLE)
    link_directories("Extern/OpenGL/lib/Apple/X64")
elseif(UNIX)
    link_directories("Extern/OpenGL/lib/Linux/X64")
    link_libraries(freeglut GL GLU GLEW)
    link_directories("Extern/irrLicht/lib/linux64")              
    link_libraries(Irrlicht libX11.so libXxf86vm.so.1)
endif()

file(COPY "Media/" DESTINATION "Media")
file(COPY "Engine/Rendering/Shader/" DESTINATION "Shader")

add_subdirectory(Engine)
add_subdirectory(Projects)


