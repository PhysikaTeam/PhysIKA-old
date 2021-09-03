file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
    add_library(ZJU_GEOMETRY STATIC ${source})
else ()
    cuda_add_library(ZJU_GEOMETRY STATIC ${source})
endif(${APPLE})


TARGET_INCLUDE_DIRECTORIES(ZJU_GEOMETRY PUBLIC 
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Geometry/Tetrahedron/Predicates/>
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Geometry/Tetrahedron/Tetrahedron/>
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Geometry/Tetrahedron/ManifoldChecking/>
  $<BUILD_INTERFACE:${ZJU_ROOT_DIR}/Geometry/Tetrahedron/CutCellGeneration/>
  )
TARGET_LINK_LIBRARIES(ZJU_GEOMETRY
cut tet pred wm_check
)
install_as_LIB(ZJU_GEOMETRY)

add_subdirectory(Tetrahedron)