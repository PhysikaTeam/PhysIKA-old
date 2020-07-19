file(GLOB header *.h *.cuh)
file(GLOB source *.cc *.c *.cpp *.cu)


if (${APPLE})
  add_library(ZJU_SOLVER STATIC ${source})
else ()
  cuda_add_library(ZJU_SOLVER STATIC ${source})
endif(${APPLE})

target_include_directories(ZJU_SOLVER PUBLIC ${CUDA_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(ZJU_SOLVER
  ZJU_CORE)
TARGET_LINK_LIBRARIES(ZJU_SOLVER
  ZJU_LINEAR_SOLVER
  ${LAPACK_LIBRARIES})

add_subdirectory(linear_solver)
