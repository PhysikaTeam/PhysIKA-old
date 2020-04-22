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


link_libraries(Core Framework IO Rendering)
link_libraries(GlutGUI)
link_libraries(ParticleSystem RigidBody)

if(WIN32)                                                               
    link_directories("${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64")
    link_libraries(freeglut glew32)            
elseif(UNIX)
    link_directories("${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Linux/X64")
    link_libraries(freeglut GLEW GL GLU)            
endif()

if(WIN32)                                                                                                   #copy dll 文件
    file(COPY "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/freeglut.dll" 
      "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/glew32.dll" 
      DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)

	  file(COPY "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/freeglut.dll" 
      "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/glew32.dll" 
      DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
endif()

file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

pybind11_add_module(${PyBind_Name} ${PY_ALL_SOURCE_FILES})

set_target_properties(${PyBind_Name} PROPERTIES
    OUTPUT_NAME "${PyBind_Name}-${PHYSIKA_LIBRARY_VERSION}")