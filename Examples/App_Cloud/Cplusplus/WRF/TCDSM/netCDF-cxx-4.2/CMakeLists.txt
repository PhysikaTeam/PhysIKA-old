project(netCDFCxx-4.2)
set(CMAKE_CXX_STANDARD 11)

set(SRC_LIST
    ${PROJECT_SOURCE_DIR}/netcdf.cpp
    ${PROJECT_SOURCE_DIR}/netcdfcpp.h
    ${PROJECT_SOURCE_DIR}/ncvalues.cpp
    ${PROJECT_SOURCE_DIR}/ncvalues.h
)

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}WRF")

target_link_libraries(
    ${PROJECT_NAME}
    ${CLOUD_netCDF_LIBS}
)


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
