file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
    add_library(ZJU_IO STATIC ${source})
else ()
    cuda_add_library(ZJU_IO STATIC ${source})
endif(${APPLE})
install_as_LIB(ZJU_IO)