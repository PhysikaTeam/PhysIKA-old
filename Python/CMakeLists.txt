cmake_minimum_required(VERSION 3.10)

add_subdirectory(${PROJECT_SOURCE_DIR}/Extern/pybind11 pybind11)

set(PyBind_Name PyPhysIKA)

file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.c*" "*.h*")



if(WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
    else()
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
    endif()
endif()   


include_directories("${CMAKE_SOURCE_DIR}/Source")                                                              #默认为本项目下各编译目标添加Source/ 目录

# temporary solution to solve linux compilation problem. our header file path should begin with Core this level
include_directories(
    "${CMAKE_SOURCE_DIR}/Source/Core" 
    "${CMAKE_SOURCE_DIR}/Source/Dynamics" 
    "${CMAKE_SOURCE_DIR}/Source/Framework" 
    "${CMAKE_SOURCE_DIR}/Source/GUI" 
    "${CMAKE_SOURCE_DIR}/Source/IO" 
    "${CMAKE_SOURCE_DIR}/Source/Rendering")

include_directories("${CMAKE_SOURCE_DIR}/Extern")                                           #默认为本项目下各编译目标添加Source/Physika_Dependency/ 目录
include_directories("${CMAKE_SOURCE_DIR}/Extern/OpenGL")                                    #默认为本项目下各编译目标添加Source/Physika_Dependency/OpenGL 目录
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})   

link_libraries(Core Framework IO Rendering)
link_libraries(GlutGUI)
link_libraries(ParticleSystem RigidBody)

file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

pybind11_add_module(${PyBind_Name} ${PY_ALL_SOURCE_FILES})

set_target_properties(${PyBind_Name} PROPERTIES
    OUTPUT_NAME "${PyBind_Name}-${PHYSIKA_LIBRARY_VERSION}")

install(TARGETS ${PyBind_Name}
    RUNTIME  DESTINATION  ${PHYSIKA_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PHYSIKA_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PHYSIKA_ARCHIVE_INSTALL_DIR}
    )