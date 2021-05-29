file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)

# if (${APPLE})
#     add_library(ZJU_CORE STATIC ${source})
# else ()
#     cuda_add_library(ZJU_CORE STATIC ${source})
# endif(${APPLE})

add_library(ZJU_CORE STATIC ${source})
