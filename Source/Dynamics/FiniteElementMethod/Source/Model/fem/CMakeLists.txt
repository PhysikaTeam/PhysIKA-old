file(GLOB header *.h)
file(GLOB source *.cc *.c)
if (${APPLE})
  add_library(ZJU_FEM SHARED ${source})
else ()
  cuda_add_library(ZJU_FEM SHARED ${source})
endif (${APPLE})

target_link_libraries(ZJU_FEM
  ZJU_CORE
 )



