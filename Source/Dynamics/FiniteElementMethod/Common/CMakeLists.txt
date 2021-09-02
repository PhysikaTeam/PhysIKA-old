file(GLOB header *.h)
file(GLOB source *.cc *.c *.cpp *.cu)



install(FILES ${header}  DESTINATION ${PHYSIKA_INC_INSTALL_DIR}/Common)
# if (${APPLE})
#     add_library(ZJU_CORE STATIC ${source})
# else ()
#     cuda_add_library(ZJU_CORE STATIC ${source})
# endif(${APPLE})

add_library(ZJU_CORE STATIC ${source})
target_include_directories(ZJU_CORE PUBLIC
$<BUILD_INTERFACE:${ZJU_ROOT_DIR}>
)

install_as_LIB(ZJU_CORE)

