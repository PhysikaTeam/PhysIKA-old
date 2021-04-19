project(tcdsmModeler)
cmake_minimum_required(VERSION 3.12)

SET(TARGET_H
        export.h
        config.h
        modeler.h
        simplemodeler.h
)

SET(TARGET_SRC
        modeler.cpp
        simplemodeler.cpp
)


if(CLOUD_PYTHON)
	SET(CMAKE_DEBUG_POSTFIX "")
    list(APPEND TARGET_SRC Gen2py.cpp Gen2py_helper.cpp)
    list(APPEND TARGET_H Gen2py_helper.h)
    pybind11_add_module(${PROJECT_NAME} ${TARGET_H} ${TARGET_SRC})
else()
    add_library(${PROJECT_NAME} SHARED ${TARGET_H} ${TARGET_SRC})
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}WRF/TCDSM")


target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE tcdsmModel
    PRIVATE COPY_DEPENDENCE
)


if(CLOUD_PYTHON AND WIN32)
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/tcdsmModeld.dll> 
                    $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/tcdsmModel.dll>
                    ${CLOUD_BINARY_DIR}/Python/TCDSM/xml
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/tcdsmUtild.dll> 
                    $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/tcdsmUtil.dll>
                    ${CLOUD_BINARY_DIR}/Python/TCDSM/xml
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CLOUD_netCDF_DLLS}
                    ${CLOUD_BINARY_DIR}/Python/TCDSM/xml
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CLOUD_OSG_DLLS}
                    ${CLOUD_BINARY_DIR}/Python/TCDSM/xml
    )
endif()


if(CLOUD_PYTHON)
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CLOUD_BINARY_DIR}/Python/TCDSM/xml
    )
else()
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        ARCHIVE DESTINATION lib/Debug
        LIBRARY DESTINATION bin/Debug
        RUNTIME DESTINATION bin/Debug
    )
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Release
        ARCHIVE DESTINATION lib/Release
        LIBRARY DESTINATION bin/Release
        RUNTIME DESTINATION bin/Release
    )
endif()