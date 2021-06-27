file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
    add_library(ZJU_GEOMETRY STATIC ${source})
else ()
    cuda_add_library(ZJU_GEOMETRY STATIC ${source})
endif(${APPLE})
install_as_LIB(ZJU_GEOMETRY)
