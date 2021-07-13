cmake_minimum_required(VERSION 3.10)

add_subdirectory(FiniteElementMethod)
set(LIB_NAMES ParticleSystem RigidBody HeightField EmbeddedMethod FastMultiphaseSPH Sand)

foreach(LIB_NAME IN ITEMS ${LIB_NAMES})
    set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}")

    file(                                                                           
        GLOB_RECURSE LIB_SRC                                                        #利用glob命令读取所有源文件list
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${LIB_SRC_DIR}/*.c*"
        "${LIB_SRC_DIR}/*.h*"
    )

    list(FILTER LIB_SRC EXCLUDE REGEX .*deprecated/.*)                              #排除deprecated 目录下面的所有文件

    if(WIN32)
        add_library(${LIB_NAME} STATIC ${LIB_SRC})                                      #添加编译目标 静态链接库
    elseif(UNIX)
        add_library(${LIB_NAME} SHARED ${LIB_SRC}) 
    endif()

    foreach(SRC IN ITEMS ${LIB_SRC})                                                #为VS工程添加filter 方便查看文件结构目录
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
    if(${LIB_NAME} STREQUAL "EmbeddedMethod")
      set(MARVEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FiniteElementMethod)
        #target_include_directories(${LIB_NAME} PUBLIC ${MARVEL_DIR}/Source ${MARVEL_DIR}/Extern)
        target_include_directories(${LIB_NAME} PUBLIC
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}/Dynamics/FiniteElementMethod/Source>
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}/Dynamics/FiniteElementMethod/Extern>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Extern/glm-0.9.9.7>
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}/Extern/glm-0.9.9.7>
        )
        #target_link_libraries(${LIB_NAME}
        #$<INSTALL_INTERFACE:ZJU_SOLVER>)
        #target_link_libraries(${LIB_NAME} ${ZJU_LIBRARIES})
        target_link_libraries(${LIB_NAME} ZJU_SOLVER embedded_elas_fem_pb elas_fem_pb embedded_mass_spring_pb mass_spring_pb)
        #starget_link_libraries(${LIB_NAME} ZJU_SOLVER embedded_elas_fem_pb elas_fem_pb embedded_mass_spring_pb mass_spring_pb)
    else()
        target_link_libraries(${LIB_NAME} PUBLIC Core Framework IO)
    endif()
    
    if(WIN32)
        target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")               #禁止编译时报告文件编码不是unicode的warning，由于cuda头文件都不是unicode。使编译报错更清晰
    endif()
    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                 #判断当前project在根目录下的相对路径
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine/Dynamics")      #为project设定folder目录
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)    #cmake 会自动延迟cuda dlink操作直到静态库被编译成动态库或者exe，
                                                                                    #但这样影响了后面其他工程对PhysLeo.lib的引用，现在还有些原因未明，暂时在编译成库之前开启dlink

    set_target_properties(${LIB_NAME} PROPERTIES
        OUTPUT_NAME "Pa${LIB_NAME}-${PHYSIKA_LIBRARY_VERSION}")
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

    target_include_directories(${LIB_NAME} PUBLIC 
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}>
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}/Dynamics>
        $<INSTALL_INTERFACE:${PHYSIKA_INC_INSTALL_DIR}/Dynamics/${LIB_NAME}>)



    install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}Targets
        RUNTIME  DESTINATION  ${PHYSIKA_RUNTIME_INSTALL_DIR}
        LIBRARY  DESTINATION  ${PHYSIKA_LIBRARY_INSTALL_DIR}
        ARCHIVE  DESTINATION  ${PHYSIKA_ARCHIVE_INSTALL_DIR}
        )

    install(EXPORT ${LIB_NAME}Targets DESTINATION ${PHYSIKA_CMAKE_CONFIG_INSTALL_DIR}
        FILE ${LIB_NAME}Targets.cmake)

    #Append ${LIB_NAME}Targets.cmake to the global list, which will be include in PhysIKAConfig.cmake
    get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
    list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
    set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

    file(GLOB FILE_DYNAMICS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}/*.h")
    install(FILES ${FILE_DYNAMICS_HEADER}  DESTINATION ${PHYSIKA_INC_INSTALL_DIR}/Dynamics/${LIB_NAME})
 endforeach()