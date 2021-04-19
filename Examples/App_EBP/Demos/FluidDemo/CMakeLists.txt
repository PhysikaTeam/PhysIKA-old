PROJECT(FluidEvolution LANGUAGES CXX CUDA)


set(SIMULATION_LINK_LIBRARIES AntTweakBar glew PositionBasedDynamics Simulation Utils)
set(SIMULATION_DEPENDENCIES AntTweakBar glew PositionBasedDynamics Simulation Utils)

if(WIN32)
  set(SIMULATION_LINK_LIBRARIES freeglut opengl32.lib glu32.lib ${SIMULATION_LINK_LIBRARIES})
  set(SIMULATION_DEPENDENCIES freeglut ${SIMULATION_DEPENDENCIES})
else()
  find_package(GLUT REQUIRED)
  find_package(OpenGL REQUIRED)

  set(SIMULATION_LINK_LIBRARIES
    ${SIMULATION_LINK_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
  )
endif()

############################################################
# Discregrid------zzl
############################################################
#include_directories(${ExternalInstallDir}/Discregrid/include)
#set(FluidDemo_DEPENDENCIES ${FluidDemo_DEPENDENCIES} ExternalProject_Discregrid)
#add_dependencies(FluidDemo ${FluidDemo_DEPENDENCIES})

include_directories(
	.
	${EigenDir} 
  ${GlewDir}
  ${GlutDir}
	${PROJECT_PATH}/extern/Discregrid/discregrid/include
	)

set(SIMULATION_DEPENDENCIES ${SIMULATION_DEPENDENCIES} ExternalProject_Discregrid)
set(SIMULATION_LINK_LIBRARIES ${SIMULATION_LINK_LIBRARIES} 
	optimized Discregrid 
	debug Discregrid_d)
link_directories(${ExternalInstallDir}/Discregrid/lib)
############################################################
# Discregrid------zzl
############################################################

file(                                                                           #利用glob命令读取所有源文件list
    GLOB_RECURSE SRC_LIST 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/*.cpp"
    "${PROJECT_SOURCE_DIR}/*.h*"
)

file(                                                                           #利用glob命令读取所有源文件list
    GLOB_RECURSE SRC_CUDA_LIST 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/*.cu*"
)

if(${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_PATH})
  list(REMOVE_ITEM SRC_LIST ${PROJECT_SOURCE_DIR}/Gen2py.cpp)
  add_executable(${PROJECT_NAME}
      ${SRC_LIST}
      ${SRC_CUDA_LIST}
      ${VIS_FILES}          
      ${PROJECT_PATH}/Common/Common.h
  )
else()
  pybind11_add_module(${PROJECT_NAME}           
          ${SRC_LIST} 
          ${SRC_CUDA_LIST}
          ${VIS_FILES}          
          ${PROJECT_PATH}/Common/Common.h)

  # set_target_properties(${PROJECT_NAME} PROPERTIES
  #   POSITION_INDEPENDENT_CODE ON
  #   CXX_VISIBILITY_PRESET "hidden"
  #   CUDA_VISIBILITY_PRESET "hidden"
  #   INTERPROCEDURAL_OPTIMIZATION TRUE
  # )
endif()

add_definitions(-DTW_NO_LIB_PRAGMA -DTW_STATIC -DFREEGLUT_LIB_PRAGMAS=0)

set_target_properties(FluidEvolution PROPERTIES FOLDER "${FOLDER_PREFIX}FluidEvolution")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_dependencies(${PROJECT_NAME} ${SIMULATION_DEPENDENCIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${SIMULATION_LINK_LIBRARIES})
