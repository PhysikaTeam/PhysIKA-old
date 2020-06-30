file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

if (${APPLE})
    add_library(ZJU_IO SHARED ${source})
else ()
    cuda_add_library(ZJU_IO SHARED ${source})
endif(${APPLE})
