project(CumulusExample)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples")

target_link_libraries(
    ${PROJECT_NAME}
    SatImageCumulus
)

if(WIN32)
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    ${cloud_opencv_shared_files_list} 
                    $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug> 
                    $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${cloud_gmp_shared_files_list}
                    $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug> 
                    $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release>
    )
endif()

install(
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Debug
    RUNTIME DESTINATION bin/Debug
)
install(
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin/Release
)