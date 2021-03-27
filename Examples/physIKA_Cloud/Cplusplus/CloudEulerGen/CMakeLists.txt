project(Cloud_Gen_Based_Euler)
set(CMAKE_CXX_STANDARD 11)

aux_source_directory(. SRC_LIST)

if(CLOUD_PYTHON)
    pybind11_add_module(${PROJECT_NAME} ${SRC_LIST})
else()
    list(REMOVE_ITEM SRC_LIST Gen2py.cpp)
    add_library(${PROJECT_NAME} ${SRC_LIST})
endif()

if(CLOUD_PYTHON)
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CLOUD_BINARY_DIR}/Python/Cloud_Gen_Based_Euler
    )
else()
    install(
        TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
    )
endif()