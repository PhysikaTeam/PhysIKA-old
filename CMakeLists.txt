cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(PyPhysLeo LANGUAGES CXX CUDA)                                                           #指定本项目的编译语言为C++、CUDA
find_package(CUDA)										                                        #目前CMake对cuda的语言支持，并没有加入原有的一些变量，所以此处仍然需要find_package来提供这些变量的值

set(CMAKE_CXX_STANDARD 14 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++14语言特性
set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")                                         #默认为所有cuda代码生成computability6.0下二进制代码

include_directories("Physika_Src")                                                              #默认为本项目下各编译目标添加Physika_Src/ 目录
include_directories("Physika_Src/Physika_Dependency")                                           #默认为本项目下各编译目标添加Physika_Src/Physika_Dependency/ 目录
include_directories("Physika_Src/Physika_Dependency/OpenGL")                                    #默认为本项目下各编译目标添加Physika_Src/Physika_Dependency/OpenGL 目录
include_directories(${CUDA_TOOLKIT_INCLUDE})                                 	                #默认为本项目下各编译目标添加cuda/include 目录
message(STATUS "cuda include directoy is ${CUDA_TOOLKIT_INCLUDE}" )

#link_libraries(${CUDA_CUDART_LIBRARY})                                                         #因为自动会添加cudart_static.lib所以不需要自己再添加别的了
#message(STATUS "cuda libraries is ${CUDA_CUDART_LIBRARY}")

if(WIN32)                                                                                       #根据不同平台添加opengl库目录和对应的链接库
    link_directories("DependentLibraries/lib/Windows/X64")
    link_libraries(freeglut glew32 glu32 opengl32)
endif()
if(APPLE)
    link_directories("DependentLibraries/lib/Apple/X64")
elseif(UNIX)
    link_directories("DependentLibraries/lib/Linux/X64")
endif()


add_subdirectory(Physika_Src)


