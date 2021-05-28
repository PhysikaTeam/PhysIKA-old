aux_source_directory(. sources)
add_library(constraint ${sources})
target_include_directories(constraint PUBLIC
$<BUILD_INTERFACE:${ZJU_ROOT_DIR}>
)