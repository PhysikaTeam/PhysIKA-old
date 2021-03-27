# Simple Tropical cyclone Modeler
# 简单的建模方法

project(STCMR)

SET(TARGET_SRC main.cpp )

add_executable(
        ${PROJECT_NAME}
        ${TARGET_SRC}
)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}Examples")

target_link_libraries(
        ${PROJECT_NAME}
        # PRIVATE ${cloud_opencv_libs}
        PRIVATE ${CLOUD_OSG_LIBS}
        PRIVATE netCDFCxx-4.2    
        PRIVATE tcdsmUtil
        PRIVATE tcdsmModel
        PRIVATE tcdsmModeler        
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
                    ${CLOUD_netCDF_DLLS}
                    $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug> 
                    $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CLOUD_OSG_DLLS}
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