project(App_CloudsOverOcean)


set_property(GLOBAL PROPERTY USE_FOLDERS ON) 
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17") 

IF(MSVC)
    SET( CMAKE_DEBUG_POSTFIX "_d" )
ENDIF(MSVC)

set(OCEAN_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(OCEAN_BINARY_DIR ${PROJECT_BINARY_DIR})


if(${CMAKE_SOURCE_DIR} STREQUAL ${OCEAN_SOURCE_DIR})
    set(FOLDER_PREFIX "")

    add_subdirectory(${OCEAN_SOURCE_DIR}/Extern)
    include_directories(
        ${OCEAN_SOURCE_DIR}/Extern
        ${OCEAN_SOURCE_DIR}/Extern/AntTweakBar/include
        ${OCEAN_SOURCE_DIR}/Extern/freeglut/include
        ${OCEAN_SOURCE_DIR}/Extern/freeglut/include/GL
        ${OCEAN_SOURCE_DIR}/Extern/glew/include
        ${OCEAN_SOURCE_DIR}/Extern/glew/include/GL
    )

    set(CMAKE_INSTALL_PREFIX ${OCEAN_BINARY_DIR}/install CACHE PATH ${OCEAN_BINARY_DIR} FORCE)

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OCEAN_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OCEAN_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OCEAN_BINARY_DIR}/bin)
else()
    set(FOLDER_PREFIX "Examples")

    add_subdirectory(${OCEAN_SOURCE_DIR}/Extern/AntTweakBar)
    include_directories(
        ${OCEAN_SOURCE_DIR}/Extern
        ${OCEAN_SOURCE_DIR}/Extern/AntTweakBar/include
        ${CMAKE_SOURCE_DIR}/Extern/freeglut-3.2.1/include
        ${CMAKE_SOURCE_DIR}/Extern/freeglut-3.2.1/include/GL
        ${CMAKE_SOURCE_DIR}/Extern/glew-1.10.0/include
        ${CMAKE_SOURCE_DIR}/Extern/glew-1.10.0/include/GL
    )

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

include_directories(
    ${OCEAN_SOURCE_DIR}
    ${OCEAN_SOURCE_DIR}/Shader
    ${OCEAN_SOURCE_DIR}/data
    ${OCEAN_SOURCE_DIR}/input
)

add_library(Ocean_DevIL UNKNOWN IMPORTED)
set_target_properties(Ocean_DevIL PROPERTIES IMPORTED_LOCATION "${OCEAN_SOURCE_DIR}/Extern/IL/DLLs/DevIL.lib")

add_library(OceanInterface INTERFACE)
target_link_libraries(OceanInterface
    INTERFACE Ocean_DevIL
    # INTERFACE Ocean_AntTweakBar
)

set(Ocean_INTERFACE_SHARED_LISTS "")
list(APPEND Ocean_INTERFACE_SHARED_LISTS "${OCEAN_SOURCE_DIR}/Extern/IL/DLLs/DevIL.dll")

add_subdirectory(${OCEAN_SOURCE_DIR}/Source)