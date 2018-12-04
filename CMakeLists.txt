cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(Physika LANGUAGES CXX CUDA)                                                             #指定本项目的编译语言为C++、CUDA
find_package(CUDA REQUIRED)										                                #目前CMake对cuda的语言支持，并没有加入原有的一些变量，所以此处仍然需要find_package来提供这些变量的值
find_package(Qt5 COMPONENTS Core Widgets)                                                       #查找qt包,可选项

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性
set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")                                         #默认为所有cuda代码生成computability3.0下二进制代码

include_directories("Physika_Src")                                                              #默认为本项目下各编译目标添加Physika_Src/ 目录
include_directories("Physika_Src/Physika_Dependency")                                           #默认为本项目下各编译目标添加Physika_Src/Physika_Dependency/ 目录
include_directories("Physika_Src/Physika_Dependency/OpenGL")                                    #默认为本项目下各编译目标添加Physika_Src/Physika_Dependency/OpenGL 目录
include_directories("Physika_Src/Physika_Dependency/irrLicht/include")                          #默认为本项目下各编译目标添加Physika_Src/Physika_dependency/irrLicht/include 目录
include_directories(${CUDA_TOOLKIT_INCLUDE})                                 	                #默认为本项目下各编译目标添加cuda/include 目录

if(Qt5_FOUND)
    link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #设置链接至QT库
endif()

link_directories("Physika_Src/Physika_Dependency/irrLicht/lib/Win64-visualStudio")              #添加irrlicht库目录和对应的库
link_libraries("Irrlicht")

if(WIN32)                                                                                       #根据不同平台添加opengl库目录和对应的链接库
    link_directories("Physika_Src/Physika_Dependency/OpenGL/lib/Windows/X64")
    link_libraries(freeglut glew32 glu32 opengl32)
endif()
if(APPLE)
    link_directories("Physika_Src/Physika_Dependency/OpenGL/lib/Apple/X64")
elseif(UNIX)
    link_directories("Physika_Src/Physika_Dependency/OpenGL/lib/Linux/X64")
endif()

file(COPY "Media/" DESTINATION "Media")

add_subdirectory(Physika_Src)
add_subdirectory(Projects)


