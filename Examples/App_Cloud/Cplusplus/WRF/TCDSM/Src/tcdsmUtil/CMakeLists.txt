project(tcdsmUtil)

SET(TARGET_H
        export.h
        config.h
        util.h
        ScopedLog.h
)

SET(TARGET_SRC
        util.cpp
        ScopedLog.cpp
)

add_library(${PROJECT_NAME} SHARED ${TARGET_H} ${TARGET_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${FOLDER_PREFIX}WRF/TCDSM")

if(USE_PREBUILT_OSG)
    target_link_libraries(
        ${PROJECT_NAME}
        ${CLOUD_OSG_LIBS}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        CLOUD_OSG
    )
endif()



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