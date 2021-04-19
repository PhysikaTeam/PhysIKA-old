PROJECT(SatImageTyphoon)
cmake_minimum_required(VERSION 3.12)


aux_source_directory(. SRC_LIST)

if(CLOUD_PYTHON)
	SET(CMAKE_DEBUG_POSTFIX "")
    pybind11_add_module(${PROJECT_NAME} ${SRC_LIST})
else()
    list(REMOVE_ITEM SRC_LIST ./Gen2py.cpp)
    add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}SatImage")


include_directories(
    ${SatImage_SOURCE_DIR}
)

target_link_libraries(
    ${PROJECT_NAME} 
    PRIVATE SatImageUtility
    PRIVATE CLOUD_OpenCV
    PRIVATE COPY_DEPENDENCE
)



if(CLOUD_PYTHON)
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CLOUD_BINARY_DIR}/Python/Typhoon/xml
    )
else()
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        ARCHIVE DESTINATION Debug/lib
    )
    install(
        TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Release
        ARCHIVE DESTINATION Release/lib
    )

    set(HEADER_FILES global.h SatDataCloud_Lite.h)
    install(
        FILES ${HEADER_FILES}
        DESTINATION include/Typhoon
    )
endif()