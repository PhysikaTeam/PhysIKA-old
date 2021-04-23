option(CLOUD_IMG_CUMULUS "option for image cumulus" ON)
if(CLOUD_IMG_CUMULUS)
    add_subdirectory(./Cumulus)
endif()