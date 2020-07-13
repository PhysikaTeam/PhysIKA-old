add_library(elas_fem_pb elas_fem_problem.cc)
target_link_libraries(elas_fem_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  )

add_library(embedded_elas_fem_pb embedded_elas_fem_problem.cc)
target_link_libraries(embedded_elas_fem_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  )

FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)


cuda_add_library(mass_spring_pb mass_spring_problem.cc)
target_link_libraries(mass_spring_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ms_hg)
TARGET_INCLUDE_DIRECTORIES(mass_spring_pb PUBLIC ${CMAKE_SOURCE_DIR}/src)


cuda_add_library(embedded_mass_spring_pb embedded_mass_spring_problem.cc)
target_link_libraries(embedded_mass_spring_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ms_hg)
TARGET_INCLUDE_DIRECTORIES(embedded_mass_spring_pb PUBLIC ${CMAKE_SOURCE_DIR}/src)



