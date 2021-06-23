add_library(elas_fem_pb elas_fem_problem.cc)
target_link_libraries(elas_fem_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ZJU_GEOMETRY)
TARGET_INCLUDE_DIRECTORIES(elas_fem_pb PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
install_as_LIB(elas_fem_pb)
add_library(embedded_elas_fem_pb embedded_elas_fem_problem.cc)
target_link_libraries(embedded_elas_fem_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ZJU_GEOMETRY)
TARGET_INCLUDE_DIRECTORIES(embedded_elas_fem_pb PUBLIC 
$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
install_as_LIB(embedded_elas_fem_pb)


add_library(mass_spring_pb mass_spring_problem.cc)
target_link_libraries(mass_spring_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ZJU_GEOMETRY
  ms_hg)
TARGET_INCLUDE_DIRECTORIES(mass_spring_pb PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
install_as_LIB(mass_spring_pb)

add_library(embedded_mass_spring_pb embedded_mass_spring_problem.cc fast_ms_problem.cc)
target_link_libraries(embedded_mass_spring_pb
  ZJU_CORE
  ZJU_FEM
  ZJU_IO
  ZJU_ENERGY
  ZJU_GEOMETRY
  ms_hg)
TARGET_INCLUDE_DIRECTORIES(embedded_mass_spring_pb PUBLIC $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)
install_as_LIB(embedded_mass_spring_pb)
