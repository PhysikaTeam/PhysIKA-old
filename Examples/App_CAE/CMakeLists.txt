cmake_minimum_required(VERSION 3.10)

if(PhysIKA_Qt5_GUI)

    set(PROJECT_NAME App_CAE)

    link_directories("${PROJECT_SOURCE_DIR}/Source")                                                           # establish link dir to Source

    if(Qt5_FOUND)
        message("Find QT5")
        link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #link Qt5::Widgets Qt5::Core Qt5::Gui
    endif()

    set(SRC_DIR "${PROJECT_SOURCE_DIR}/Examples/${PROJECT_NAME}")

    file(                                                                                                       #get src files by glob
        GLOB_RECURSE SRC_LIST
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${SRC_DIR}/*.c*"
        "${SRC_DIR}/*.h*"
    )

    list(FILTER SRC_LIST EXCLUDE REGEX .*Media/.*)                                                              #Exclude resource files

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 #prepare for building

    if(WIN32)
        target_link_libraries(${PROJECT_NAME} PUBLIC
            Core
            Framework
            IO
            ParticleSystem
            RigidBody
            QtGUI)


        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG           #/WHOLEARCHIVE is used to link unused static variables for visual studio, which is necessary for auto class registration.
           "/WHOLEARCHIVE:PaFramework-${PHYSIKA_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:PaParticleSystem-${PHYSIKA_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:PaIO-${PHYSIKA_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:PaQtGUI-${PHYSIKA_LIBRARY_VERSION}d.lib")

        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE
           "/WHOLEARCHIVE:PaFramework-${PHYSIKA_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:PaParticleSystem-${PHYSIKA_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:PaIO-${PHYSIKA_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:PaQtGUI-${PHYSIKA_LIBRARY_VERSION}.lib")
    else()
        message(WARNING "Could not set install folders for this platform!")
    endif()

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #get relative path
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples")                              #set target properties
    set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    foreach(SRC IN ITEMS ${SRC_LIST})                                                                           #traverse source files
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
#else()
#   message(FATAL_ERROR "PhysIKA_Qt5_GUI failed and App_CAE won't be build!")
endif()

