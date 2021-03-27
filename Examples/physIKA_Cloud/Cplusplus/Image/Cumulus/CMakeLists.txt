project(Sim_Cloud)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-D_USE_MATH_DEFINES) 


aux_source_directory(. SRC_LIST)

if(CLOUD_PYTHON)
	SET(CMAKE_DEBUG_POSTFIX "")
    pybind11_add_module(${PROJECT_NAME} ${SRC_LIST})
else()
    list(REMOVE_ITEM SRC_LIST ./Gen2py.cpp)
    add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Image")


target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE CLOUD_OpenMesh
    PRIVATE CLOUD_OpenCV
    PRIVATE CLOUD_GMP
    PRIVATE COPY_DEPENDENCE
)


if(WIN32)
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    ${cloud_opencv_shared_files_list} 
                    ${CLOUD_BINARY_DIR}/Python/ImageCumulus/xml
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${cloud_gmp_shared_files_list}
                    ${CLOUD_BINARY_DIR}/Python/ImageCumulus/xml
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different
        #             ${PROJECT_NAME}
        #             ${CLOUD_BINARY_DIR}/Python/ImageCumulus/xml
    )
elseif(UNIX)
    message("UNIX")
endif()


if(CLOUD_PYTHON)
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CLOUD_BINARY_DIR}/Python/ImageCumulus/xml
    )
else()
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        ARCHIVE DESTINATION lib/Debug
    )
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Release
        ARCHIVE DESTINATION lib/Release
    )
endif()

