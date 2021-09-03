file(GLOB header *.h *.cuh)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
  add_library(ZJU_LINEAR_SOLVER STATIC ${source})
else ()
  cuda_add_library(ZJU_LINEAR_SOLVER STATIC ${source})
endif(${APPLE})

target_include_directories(ZJU_LINEAR_SOLVER PUBLIC
$<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>
$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)
target_link_libraries(ZJU_LINEAR_SOLVER
  ZJU_CORE)
TARGET_LINK_LIBRARIES(ZJU_LINEAR_SOLVER ${LAPACK_LIBRARIES})
install_as_LIB(ZJU_LINEAR_SOLVER)