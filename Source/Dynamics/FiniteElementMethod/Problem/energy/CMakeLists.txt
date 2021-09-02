file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)


if (${APPLE})
  add_library(ZJU_ENERGY STATIC ${source})
else ()
  cuda_add_library(ZJU_ENERGY STATIC ${source})
endif(${APPLE})

target_link_libraries(ZJU_ENERGY
  ZJU_CORE)

install_as_LIB(ZJU_ENERGY)