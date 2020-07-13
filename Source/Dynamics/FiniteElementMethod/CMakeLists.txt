project(MarvalPhysics)
cmake_minimum_required(VERSION 3.10)
# The project version number.
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")

if (${APPLE})
  list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -ftemplate-backtrace-limit=0 -fPIC")
else ()
  list(APPEND CMAKE_CXX_FLAGS "-fopenmp -std=c++11 -fPIC")
endif (${APPLE})

# BOOST
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS filesystem system program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# EIGEN3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (${APPLE})
  message ("MAC OS X, avoid adding cuda")
else ()
  # CUDA
  FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
  STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
  SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)
  #TODO: make this available for others
  SET(CUSTOM_CUDA_CUBLAS_LIBRARIES
    /usr/local/cuda-10.0/lib64/libcublas.so)
  IF(EXISTS ${CUSTOM_CUDA_CUBLAS_LIBRARIES})
    SET(CUDA_CUBLAS_LIBRARIES
      # ${CUDA_CUBLAS_LIBRARIES}
      ${CUSTOM_CUDA_CUBLAS_LIBRARIES})
  ENDIF()
  SET(CUDA_NVCC_FLAGS --compiler-options "-std=c++11")
  SET(CUDA_HELPER_PATH ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
  include_directories(${CUDA_HELPER_PATH})
  include_directories(${CUDA_INCLUDE_DIR})
  list(APPEND CUDA_NVCC_FLAGS "-g -G")
endif (${APPLE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Extern)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source)

add_subdirectory(Source)
add_subdirectory(Docs EXCLUDE_FROM_ALL)
