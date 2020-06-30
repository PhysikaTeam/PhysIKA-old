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

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/;${CMAKE_MODULE_PATH}")

# BOOST
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS filesystem system program_options REQUIRED)
if(Boost_FOUND)
  message("-- Boost @ ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

# EIGEN3
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  message("-- Eigen @ ${EIGEN3_INCLUDE_DIR}")
  message("-- Eigen version: ${EIGEN3_VERSION}")
  include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

if (${APPLE})
  message ("MAC OS X, avoid adding cuda")
else ()
  # CUDA
  FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
  STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
  SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)
  #TODO: make this available for others
  SET(CUSTOM_CUDA_CUBLAS_LIBRARIES
    # /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs/libcublas.so
    /usr/local/cuda-10.0/lib64/libcublas.so)
  # SET(CUSTOM_CUDA_cusparse_LIBRARY
  #   # /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusparse_static.a
  #   /usr/local/cuda-10.0/lib64/libcusparse.so)
  IF(EXISTS ${CUSTOM_CUDA_CUBLAS_LIBRARIES})
    SET(CUDA_CUBLAS_LIBRARIES
      # ${CUDA_CUBLAS_LIBRARIES}
      ${CUSTOM_CUDA_CUBLAS_LIBRARIES})
  ENDIF()
  # IF(EXISTS ${CUSTOM_CUDA_cusparse_LIBRARY})
  #   SET(CUDA_cusparse_LIBRARY
  #     # ${CUDA_cusparse_LIBRARY}
  #     ${CUSTOM_CUDA_cusparse_LIBRARY})
  # ENDIF()
  SET(CUDA_NVCC_FLAGS --compiler-options "-std=c++11")
  SET(CUDA_HELPER_PATH ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
  include_directories(${CUDA_HELPER_PATH})
  include_directories(${CUDA_INCLUDE_DIR})
  list(APPEND CUDA_NVCC_FLAGS "-g -G")
endif (${APPLE})

#TBB
find_package(TBB REQUIRED)
if(TBB_FOUND)
  message("-- TBB @ ${TBB_INCLUDE_DIRS}")
  include_directories(${TBB_INCLUDE_DIRS})
endif(TBB_FOUND)

#lapack
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
  message("Find lapack.")
endif(LAPACK_FOUND)

if (NOT ${APPLE})
  find_package(OpenMP REQUIRED)
  if(NOT TARGET OpenMP::OpenMP_CXX)
      add_library(OpenMP_TARGET INTERFACE)
      add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
      target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
      find_package(Threads REQUIRED)
      target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
      target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})

      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif (NOT ${APPLE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Extern)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Source)

add_subdirectory(Source)
add_subdirectory(Docs EXCLUDE_FROM_ALL)
