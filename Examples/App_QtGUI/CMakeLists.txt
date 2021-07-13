cmake_minimum_required(VERSION 3.10)

if(PhysIKA_Qt5_GUI)

    set(PROJECT_NAME App_QtGUI)

    link_directories("${PROJECT_SOURCE_DIR}/Source")                                                           # "y·

    if(Qt5_FOUND)
        message("Find QT5")
        link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #QT
    endif()

    set(SRC_DIR "${PROJECT_SOURCE_DIR}/Examples/${PROJECT_NAME}")

    file(                                                                                                       #globllist
        GLOB_RECURSE SRC_LIST
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${SRC_DIR}/*.c*"
        "${SRC_DIR}/*.h*"
    )

    list(FILTER SRC_LIST EXCLUDE REGEX .*Media/.*)                                                              #udeprecated ll

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 #L l

    if(UNIX)
        target_link_libraries(${PROJECT_NAME} PUBLIC                                   #the option "-Wl,--whole-archive" is an equivalance to /WHOLEARCHIVE
            -Wl,--whole-archive
            Core
            Framework
            IO
            ParticleSystem
            QtGUI
            RigidBody
            -Wl,--no-whole-archive)

    elseif(WIN32)
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
        message(FATAL_ERROR "Could not set install folders for this platform!")
    endif()

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #жεprojectL 1/4 u·
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples")                              #project趨folderL 1/4 
    set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    foreach(SRC IN ITEMS ${SRC_LIST})                                                                           #VSfilter lL 1/4 
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()

endif()

