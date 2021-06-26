file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
    add_library(ZJU_GEOMETRY STATIC ${source})
else ()
    cuda_add_library(ZJU_GEOMETRY STATIC ${source})
endif(${APPLE})


TARGET_INCLUDE_DIRECTORIES(ZJU_GEOMETRY PUBLIC 
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Tetrahedron/src/tetrahedron/inc>
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Tetrahedron/src/manifold_checking/inc/>
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Tetrahedron/src/cut_cell_generation/inc/>
  )
TARGET_LINK_LIBRARIES(ZJU_GEOMETRY
cut tet pred wm_check
)
install_as_LIB(ZJU_GEOMETRY)
