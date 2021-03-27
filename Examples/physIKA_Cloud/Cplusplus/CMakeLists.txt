include_directories(
    ./
)

option(CLOUD_EULER "option for euler method" OFF)
if(CLOUD_EULER)
    add_subdirectory(./CloudEulerGen)
endif()


add_subdirectory(./WRF)

add_subdirectory(./SatImage)

add_subdirectory(./Image)