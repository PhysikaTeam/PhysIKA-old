project(SatImageUtility)
cmake_minimum_required(VERSION 3.12)

aux_source_directory(. SRC_LIST)
add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}SatImage")

install(
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Debug
    ARCHIVE DESTINATION Debug/lib
)

set(HEADER_FILES Color.h perlin.h Vector.h)
install(
    FILES ${HEADER_FILES}
    DESTINATION include/Utility
)