
set(TARGET_H
    Camera.h
    color_encoding.h
    Color.h
    common.h
    Frustum.h
    GLText.h
    mat4.h
    perlin.h
    Program.h
    SkyDome.h
    timer.h
    vec4.h
    Vector.h
    VolumeClouds.h
)


set(TARGET_SRC
    Camera.cpp
    Frustum.cpp
    perlin.cpp
    Program.cpp
    SkyDome.cpp
    timer.cpp
    GLText.cpp
    VolumeClouds.cpp
    Main.cpp
)

add_definitions(-DPROJECT_SOURCE_DIR="${OCEAN_SOURCE_DIR}")

set(executable_name App_OceanClouds)
add_executable(${executable_name} ${TARGET_SRC} ${TARGET_H})
set_target_properties(${executable_name} PROPERTIES FOLDER "${FOLDER_PREFIX}")

target_link_libraries(
    ${executable_name}
    PRIVATE freeglut
    PRIVATE glew
    PRIVATE OceanInterface
    PRIVATE AntTweakBar
)

add_custom_command(
    TARGET ${executable_name}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                ${Ocean_INTERFACE_SHARED_LISTS} 
                $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug> 
                $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release>
)

install(
    TARGETS ${executable_name}
    CONFIGURATIONS Debug
    RUNTIME DESTINATION bin/Debug
)
install(
    TARGETS ${executable_name}
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin/Release
)