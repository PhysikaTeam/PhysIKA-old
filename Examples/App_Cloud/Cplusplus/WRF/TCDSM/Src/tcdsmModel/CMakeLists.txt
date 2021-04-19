project(tcdsmModel)

SET(TARGET_H
        export.h
        config.h
        model.h
        particles.h
        netcdfparser.h
        netcdfset.h
        netcdfclassifier.h
        wrfnetcdfparser.h
        netcdfparserbuilder.h
)
SET(TARGET_SRC
        model.cpp
        particles.cpp
        netcdfclassifier.cpp
        netcdfset.cpp
        wrfnetcdfparser.cpp
        netcdfparserbuilder.cpp
)

add_library(${PROJECT_NAME} SHARED ${TARGET_H} ${TARGET_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}WRF/TCDSM")

target_link_libraries(
    ${PROJECT_NAME}
    tcdsmUtil
    netCDFCxx-4.2
    # PRIVATE tcdsmUtil
    # ${CLOUD_OSG_LIBS}
)

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

