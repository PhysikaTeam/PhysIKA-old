PROJECT(App_Cloth2)
cmake_minimum_required(VERSION 3.12)

link_libraries(Core Framework IO Rendering)
link_libraries(ParticleSystem)
link_libraries(GlutGUI)


aux_source_directory(. SRC_LIST)
# SET(CMAKE_DEBUG_POSTFIX "")
pybind11_add_module(${PROJECT_NAME} ${SRC_LIST})

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #判断当前project在根目录下的相对路径
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples") 


if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    
    set(python_files_list
        "${PROJECT_SOURCE_DIR}/cloth.xml"
        "${PROJECT_SOURCE_DIR}/cloth_run.py"
        "${PROJECT_SOURCE_DIR}/cloth_xml_parsing.py"
    )
	
    file(COPY ${python_files_list} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
    file(COPY ${python_files_list} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()    


install(TARGETS ${PROJECT_NAME}
    RUNTIME  DESTINATION  ${PHYSIKA_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PHYSIKA_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PHYSIKA_ARCHIVE_INSTALL_DIR}
    )