cmake_minimum_required(VERSION 3.10)

add_subdirectory(${PROJECT_SOURCE_DIR}/Extern/pybind11 pybind11)

set(PyBind_Name PyPhysIKA)

file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.c*" "*.h*")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


link_libraries(Core Framework IO Rendering)
link_libraries(GlutGUI)
link_libraries(ParticleSystem RigidBody)

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

