cmake_minimum_required(VERSION 3.10)

set(PROJECTS_NAMES App_Collision App_SingleFluid App_MultipleFluid App_Elasticity App_Hyperelasticity App_Plasticity App_Cloth App_Viscoplasticity App_DrySand App_WetSand App_Fracture App_SFI App_Rod)

link_directories("${PROJECT_SOURCE_DIR}/Engine")                                                           # 设置库路径
link_libraries(Core Framework IO Rendering)
link_libraries(ParticleSystem RigidBody)
link_libraries(GlutGUI)

if(Qt5_FOUND)
    message("Find QT5")
    link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #设置链接至QT库
endif()

if(Qt5_FOUND)
    link_libraries(QtGUI)                                                                                       #链接QtGUI库
endif()   

foreach(PROJECT_NAME IN ITEMS ${PROJECTS_NAMES})
    set(SRC_DIR "${PROJECT_SOURCE_DIR}/Projects/${PROJECT_NAME}")

    file(                                                                                                       #利用glob命令读取所有源文件list
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${SRC_DIR}/*.c*"
        "${SRC_DIR}/*.h*"
    )

    list(FILTER SRC_LIST EXCLUDE REGEX .*Media/.*)                                                              #排除deprecated 文件下面的所有文件

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 #添加编译目标 可执行文件
    target_link_libraries(${PROJECT_NAME} PUBLIC Core)
    target_link_libraries(${PROJECT_NAME} PUBLIC Core Framework IO Rendering ParticleSystem RigidBody GlutGUI)

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #判断当前project在根目录下的相对路径
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${PROJECT_PATH_REL}")                              #为project设定folder目录

    if(WIN32)                                                                                                   #copy dll 文件
        file(COPY "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/freeglut.dll" 
			  "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/glew32.dll" 
			  "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/glu32.dll" 
			  "${PROJECT_SOURCE_DIR}/Extern/OpenGL/lib/Windows/X64/opengl32.dll"
              "${PROJECT_SOURCE_DIR}/Extern/irrLicht/lib/Win64-visualStudio/Irrlicht.dll" 
			  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()


    foreach(SRC IN ITEMS ${SRC_LIST})                                                                           #为VS工程添加filter 方便查看文件结构目录
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()

endforeach()
