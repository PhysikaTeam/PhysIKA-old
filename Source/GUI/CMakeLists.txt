cmake_minimum_required(VERSION 3.10)

set(LIB_NAMES GlutGUI QtGUI)

#if(NOT VTK_BINARY_DIR)
#    find_package(VTK)
#    if(NOT VTK_DIR)
#        MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
#    endif(NOT VTK_DIR)
#    include(${VTK_USE_FILE})
#endif(NOT VTK_BINARY_DIR)

if(NOT Qt5_FOUND)                                                                   #如果没有安装Qt5，那么不编译QtGUI模块
    list(REMOVE_ITEM LIB_NAMES QtGUI)
endif()

if(Qt5_FOUND)
    message("Find QT5")
    link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #设置链接至QT库
endif()

foreach(LIB_NAME IN ITEMS ${LIB_NAMES})
    set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}")

    if(${LIB_NAME} STREQUAL QtGUI)
        set(CMAKE_AUTOMOC ON)                                                                   # 开启自动moc功能
        set(CMAKE_AUTOUIC ON)                                                                   # 开启自动uic功能
    endif()

    file(                                                                           
        GLOB_RECURSE LIB_SRC                                                        #利用glob命令读取所有源文件list
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${LIB_SRC_DIR}/*.c*"
        "${LIB_SRC_DIR}/*.h*"
    )

    list(FILTER LIB_SRC EXCLUDE REGEX .*deprecated/.*)                              #排除deprecated 目录下面的所有文件

    add_library(${LIB_NAME} STATIC ${LIB_SRC})                                      #添加编译目标 静态链接库

    foreach(SRC IN ITEMS ${LIB_SRC})                                                #为VS工程添加filter 方便查看文件结构目录
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()

    if(WIN32)
        target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")               #禁止编译时报告文件编码不是unicode的warning，由于cuda头文件都不是unicode。使编译报错更清晰
    endif()
    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                 #判断当前project在根目录下的相对路径
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine/GUI")      #为project设定folder目录
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)    #cmake 会自动延迟cuda dlink操作直到静态库被编译成动态库或者exe，
                                                                                    #但这样影响了后面其他工程对PhysLeo.lib的引用，现在还有些原因未明，暂时在编译成库之前开启dlink
    
endforeach()

#target_link_libraries(QtGUI
#    ${VTK_LIBRARIES}
#)