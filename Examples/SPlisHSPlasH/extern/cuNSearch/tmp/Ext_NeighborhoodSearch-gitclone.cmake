
if(NOT "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitinfo.txt" IS_NEWER_THAN "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: 'D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "D:/Git/cmd/git.exe"  clone  "https://github.com/InteractiveComputerGraphics/cuNSearch.git" "Ext_NeighborhoodSearch"
    WORKING_DIRECTORY "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/InteractiveComputerGraphics/cuNSearch.git'")
endif()

execute_process(
  COMMAND "D:/Git/cmd/git.exe"  checkout aba3da18cb4f45cd05d729465d1725891ffc33da --
  WORKING_DIRECTORY "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'aba3da18cb4f45cd05d729465d1725891ffc33da'")
endif()

execute_process(
  COMMAND "D:/Git/cmd/git.exe"  submodule update --recursive --init 
  WORKING_DIRECTORY "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitinfo.txt"
    "D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'D:/PhysIKA_merge/Examples/SPlisHSPlasH/extern/cuNSearch/src/Ext_NeighborhoodSearch-stamp/Ext_NeighborhoodSearch-gitclone-lastrun.txt'")
endif()

