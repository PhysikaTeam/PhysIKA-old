file(GLOB header *.h *.cuh)
file(GLOB source *.cc *.c *.cpp *.cu linear_solver/*.cc linear_solver/*.cu)
if (${APPLE})
  add_library(ZJU_LINEAR_SOLVER SHARED ${source})
else ()
  cuda_add_library(ZJU_LINEAR_SOLVER SHARED ${source})
endif(${APPLE})

target_include_directories(ZJU_LINEAR_SOLVER PUBLIC ${CUDA_INCLUDE_DIR})
target_link_libraries(ZJU_LINEAR_SOLVER
  ZJU_CORE)
TARGET_LINK_LIBRARIES(ZJU_LINEAR_SOLVER
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_cusparse_LIBRARY}
  ${LAPACK_LIBRARIES}
  )
