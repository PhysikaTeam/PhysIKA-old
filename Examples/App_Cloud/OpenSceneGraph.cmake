set(CLOUD_OSG_SOURCE_DIR ${CLOUD_SOURCE_DIR}/Extern/OpenSceneGraph-3.6.5)
set(CLOUD_OSG_BINARY_DIR ${CLOUD_BINARY_DIR}/Extern/OpenSceneGraph-3.6.5/build)
set(CLOUD_OSG_LIB_DIR ${CLOUD_OSG_BINARY_DIR}/lib)
set(CLOUD_OSG_BIN_DIR ${CLOUD_OSG_BINARY_DIR}/bin)
# message("${CLOUD_OSG_BIN_DIR}")

add_library(CLOUD_OSG ${CLOUD_SOURCE_DIR}/Extern/openscenegraph_interface.cpp)
set_target_properties(CLOUD_OSG PROPERTIES FOLDER "${FOLDER_PREFIX}ExternalProjectTargets")


include_directories(
    ${CLOUD_OSG_SOURCE_DIR}/include
    ${CLOUD_OSG_BINARY_DIR}/include
)


add_custom_command(
    TARGET CLOUD_OSG
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -S ${CLOUD_OSG_SOURCE_DIR} -B ${CLOUD_OSG_BINARY_DIR} -A x64
                                -DBUILD_OSG_APPLICATIONS=OFF -DBUILD_OSG_EXAMPLES=OFF -DBUILD_OSG_PLUGINS_BY_DEFAULT=OFF
    COMMAND ${CMAKE_COMMAND} --build ${CLOUD_OSG_BINARY_DIR} --config $<$<CONFIG:>:Undefined>$<$<NOT:$<CONFIG:>>:$<CONFIG>>
                                --target osg OpenThreads
)


add_library(osg SHARED IMPORTED)
set_target_properties(osg PROPERTIES 
    IMPORTED_IMPLIB_DEBUG "${CLOUD_OSG_LIB_DIR}/osgd.lib"
    IMPORTED_LOCATION_DEBUG "${CLOUD_OSG_BIN_DIR}/osg161-osgd.dll"
)
set_target_properties(osg PROPERTIES 
    IMPORTED_IMPLIB_RELEASE "${CLOUD_OSG_LIB_DIR}/osg.lib"
    IMPORTED_LOCATION_RELEASE "${CLOUD_OSG_BIN_DIR}/osg161-osg.dll"
)


add_library(OpenThreads SHARED IMPORTED)
set_target_properties(OpenThreads PROPERTIES 
    IMPORTED_IMPLIB_DEBUG "${CLOUD_OSG_LIB_DIR}/OpenThreadsd.lib"
    IMPORTED_LOCATION_DEBUG "${CLOUD_OSG_BIN_DIR}/ot21-OpenThreadsd.dll"
)
set_target_properties(OpenThreads PROPERTIES 
    IMPORTED_IMPLIB_RELEASE "${CLOUD_OSG_LIB_DIR}/OpenThreads.lib"
    IMPORTED_LOCATION_RELEASE "${CLOUD_OSG_BIN_DIR}/ot21-OpenThreads.dll"
)

set(CLOUD_OSG_LINK_LIST osg OpenThreads)

target_link_libraries(CLOUD_OSG
    ${CLOUD_OSG_LINK_LIST}
)

set(CLOUD_OSG_DLLS "")
list(
    APPEND CLOUD_OSG_DLLS
    $<$<CONFIG:Debug>:${CLOUD_OSG_BIN_DIR}/osg161-osgd.dll> 
    $<$<CONFIG:Release>:${CLOUD_OSG_BIN_DIR}/osg161-osg.dll>
)
list(
    APPEND CLOUD_OSG_DLLS
    $<$<CONFIG:Debug>:${CLOUD_OSG_BIN_DIR}/ot21-OpenThreadsd.dll> 
    $<$<CONFIG:Release>:${CLOUD_OSG_BIN_DIR}/ot21-OpenThreads.dll>
)