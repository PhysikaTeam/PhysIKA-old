project(MarvalPhysics CXX)
set(ZJU_ROOT_DIR ${CMAKE_SOURCE_DIR}/Source/Dynamics/FiniteElementMethod/Source)
cmake_minimum_required(VERSION 3.10)
# The project version number.
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")

if (${APPLE})
  list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -ftemplate-backtrace-limit=0 -fPIC")
else ()
  list(APPEND CMAKE_CXX_FLAGS " -std=c++11 -fPIC")
endif (${APPLE})

function(install_as_LIB LIB_NAME)
install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PHYSIKA_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PHYSIKA_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PHYSIKA_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PHYSIKA_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})
endfunction()

set(CMAKE_CXX_STANDARD 14)
if (WIN32)
  add_compile_options(-bigobj)
endif ()

# find_package(OpenMP REQUIRED)
# link_libraries(OpenMP::OpenMP_CXX)
# add_definitions(-DNULLEMBEDDED)

# BOOST
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# find_package(Boost COMPONENTS filesystem system program_options REQUIRED)
# set(Boost_INCLUDE_DIRS C:/Boost/include/boost-1_73)
set(Boost_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Extern)
include_directories(${Boost_INCLUDE_DIRS})


# EIGEN3
# find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/Extern)
include_directories(${EIGEN3_INCLUDE_DIR})

if (${APPLE})
  message ("MAC OS X, avoid adding cuda")
else ()
  # CUDA
  FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
  STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
  SET(CUDA_NVCC_FLAGS --compiler-options "-std=c++11")
  SET(CUDA_HELPER_PATH ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
  include_directories(${CUDA_HELPER_PATH})
  include_directories(${CUDA_INCLUDE_DIR})
  list(APPEND CUDA_NVCC_FLAGS "-g -G")
endif (${APPLE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Extern)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source)

add_subdirectory(Source)
# add_subdirectory(Docs EXCLUDE_FROM_ALL)
