project(SatImageALL)

set(SatImage_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SatImage_BINARY_DIR ${PROJECT_BINARY_DIR})

option(CLOUD_SAT_CUMULUS "option for satimage cumulus" ON)
option(CLOUD_SAT_TYPHOON "option for satimage typhoon" ON)


if(CLOUD_SAT_CUMULUS OR CLOUD_SAT_TYPHOON)
    add_subdirectory(Utility)
endif()

if(CLOUD_SAT_CUMULUS)
    add_subdirectory(Cumulus)
endif()

if(CLOUD_SAT_TYPHOON)
    add_subdirectory(Typhoon)
endif()