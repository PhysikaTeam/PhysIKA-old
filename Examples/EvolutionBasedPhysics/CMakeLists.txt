cmake_minimum_required(VERSION 3.10)

project(EvolutionBasedPhysics)

set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_PATH})


if(${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

  set(EigenDir "${CMAKE_SOURCE_DIR}/extern/eigen")
  set(GlewDir "${PROJECT_PATH}/extern/glew/include")
  set(GlutDir "${PROJECT_PATH}/extern/freeglut/include")

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  set(CMAKE_DEBUG_POSTFIX "d")

  set(FOLDER_PREFIX "")
else()
  set(EigenDir "${CMAKE_SOURCE_DIR}/Extern")
  set(GlewDir "${CMAKE_SOURCE_DIR}/Extern/glew-1.10.0/include")
  set(GlutDir "${CMAKE_SOURCE_DIR}/Extern/freeglut-3.2.1/include")
  
  set(FOLDER_PREFIX "Examples/${PROJECT_NAME}/")
endif()

# my deleted
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
# include(${PROJECT_PATH}/CMake/Common.cmake)

add_definitions(-D_CRT_SECURE_NO_DEPRECATE)

if (WIN32)
  if(${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
    subdirs(extern/freeglut extern/AntTweakBar extern/glew extern/md5 Demos PositionBasedDynamics)
  else()
    subdirs(extern/AntTweakBar extern/md5 Demos PositionBasedDynamics)
  endif()
else()
  subdirs(extern/AntTweakBar extern/glew extern/md5 Demos PositionBasedDynamics)
endif()

include(ExternalProject)

if (DEFINED PBD_EXTERNALINSTALLDIR)
	set(ExternalInstallDir "${PBD_EXTERNALINSTALLDIR}")	
else()
	set(ExternalInstallDir "${PROJECT_BINARY_DIR}/extern/install")
endif()


## Discregrid
ExternalProject_Add(
   ExternalProject_Discregrid
   PREFIX ${PROJECT_BINARY_DIR}/extern/Discregrid
   SOURCE_DIR  ${PROJECT_PATH}/extern/Discregrid
   BINARY_DIR  ${PROJECT_BINARY_DIR}/extern
   INSTALL_DIR ${ExternalInstallDir}/Discregrid
   CMAKE_ARGS  -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/Discregrid -DBUILD_CMD_EXECUTABLE:BOOL=0 -DEIGEN3_INCLUDE_DIR:PATH=${EigenDir}
)
set_target_properties(ExternalProject_Discregrid PROPERTIES FOLDER "${FOLDER_PREFIX}External Dependencies")


# install(DIRECTORY ./Common
# DESTINATION include
# FILES_MATCHING PATTERN "*.h")

# install(DIRECTORY ./data
# DESTINATION include
# FILES_MATCHING PATTERN "*.glsl")

