cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME App_Test)

link_directories("${PROJECT_SOURCE_DIR}/Source")                                                           # 设置库路径

if(Qt5_FOUND)
    message("Find QT5")
    link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)                                             #设置链接至QT库
endif()

set(SRC_DIR "${PROJECT_SOURCE_DIR}/Examples/${PROJECT_NAME}")

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
target_link_libraries(${PROJECT_NAME} PUBLIC Core Framework IO Rendering ParticleSystem RigidBody QtGUI) 

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #判断当前project在根目录下的相对路径
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples")                              #为project设定folder目录
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

foreach(SRC IN ITEMS ${SRC_LIST})                                                                           #为VS工程添加filter 方便查看文件结构目录
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()