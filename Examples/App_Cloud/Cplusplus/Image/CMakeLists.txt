option(CLOUD_IMG_CUMULUS "option for image cumulus" OFF)
if(CLOUD_IMG_CUMULUS)
    add_subdirectory(./Cumulus)
endif()