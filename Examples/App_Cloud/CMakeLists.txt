cmake_minimum_required(VERSION 3.12)
include(ExternalProject)

project(App_Cloud)

set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

IF(MSVC)
    SET( CMAKE_DEBUG_POSTFIX "d" )
ENDIF(MSVC)


set(CLOUD_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(CLOUD_BINARY_DIR ${PROJECT_BINARY_DIR})


# 如果没在PhySIKA目录下
if(${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
    set(CMAKE_INSTALL_PREFIX ${CLOUD_BINARY_DIR}/install CACHE PATH ${CLOUD_BINARY_DIR} FORCE)

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CLOUD_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CLOUD_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CLOUD_BINARY_DIR}/bin)

    set(FOLDER_PREFIX "")

    set(PhysIKA_Source_DIR "")
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
    string(REGEX REPLACE "//" "" CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    set(FOLDER_PREFIX "Examples/Cloud/")

    set(PhysIKA_Source_DIR ${CMAKE_SOURCE_DIR}/Source)

    add_definitions(-DPhysIKA)
endif()


# import OpenCV part
set(CLOUD_OpenCV_SOURCE_DIR ${CLOUD_SOURCE_DIR}/Extern/opencv-2.4.13)
set(CLOUD_OpenCV_BINARY_DIR ${CLOUD_BINARY_DIR}/Extern/opencv-2.4.13/build)

set(CLOUD_OpenCV_INCLUDE_DIR ${CLOUD_OpenCV_SOURCE_DIR}/install-include)
set(CLOUD_OpenCV_LIB_DIR ${CLOUD_OpenCV_BINARY_DIR}/install/lib)
set(CLOUD_OpenCV_BIN_DIR ${CLOUD_OpenCV_BINARY_DIR}/install/bin)

if(UNIX)
    message("UNIX")
elseif(WIN32)
    add_library(CLOUD_OpenCV ${CLOUD_SOURCE_DIR}/Extern/opencv_interface.cpp)
    set_target_properties(CLOUD_OpenCV PROPERTIES FOLDER "${FOLDER_PREFIX}ExternalProjectTargets")

    add_custom_command(
        TARGET CLOUD_OpenCV
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -S ${CLOUD_OpenCV_SOURCE_DIR} -B ${CLOUD_OpenCV_BINARY_DIR} -DWITH_CUDA=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF -DWITH_EIGEN=OFF 
                    -DWITH_OPENCL=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_opencv_apps=OFF -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=OFF 
                    -DBUILD_opencv_features2d=OFF -DBUILD_opencv_flann=OFF -DBUILD_opencv_objdetect=OFF -DBUILD_opencv_video=OFF -DBUILD_opencv_superes=OFF 
                    -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=OFF -DWITH_FFMPEG=OFF
                    -A x64
        COMMAND ${CMAKE_COMMAND} --build ${CLOUD_OpenCV_BINARY_DIR} --config $<$<CONFIG:>:Undefined>$<$<NOT:$<CONFIG:>>:$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} --install ${CLOUD_OpenCV_BINARY_DIR} --config $<$<CONFIG:>:Undefined>$<$<NOT:$<CONFIG:>>:$<CONFIG>>
    )

    set(cloud_opencv_libs
        opencv_core
        opencv_highgui
        opencv_imgproc
    )
    set(cloud_opencv_link_list "")
    set(cloud_opencv_shared_files_list "")

    foreach(lib ${cloud_opencv_libs})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES 
            IMPORTED_IMPLIB_DEBUG "${CLOUD_OpenCV_LIB_DIR}/${lib}2413d.lib"
            IMPORTED_LOCATION_DEBUG "${CLOUD_OpenCV_BIN_DIR}/${lib}2413d.dll"
        )
        set_target_properties(${lib} PROPERTIES 
            IMPORTED_IMPLIB_RELEASE "${CLOUD_OpenCV_LIB_DIR}/${lib}2413.lib"
            IMPORTED_LOCATION_RELEASE "${CLOUD_OpenCV_BIN_DIR}/${lib}2413.dll"
        )
        list(APPEND cloud_opencv_link_list ${lib})
        list(
            APPEND cloud_opencv_shared_files_list 
            $<$<CONFIG:Debug>:${CLOUD_OpenCV_BIN_DIR}/${lib}2413d.dll> 
            $<$<CONFIG:Release>:${CLOUD_OpenCV_BIN_DIR}/${lib}2413.dll>
        )
    endforeach()

    target_link_libraries(CLOUD_OpenCV
        ${cloud_opencv_link_list}
    )
endif()



# import CGAL part
set(CLOUD_CGAL_SOURCE_DIR ${CLOUD_SOURCE_DIR}/Extern/CGAL5.1)
set(CLOUD_CGAL_BINARY_DIR ${CLOUD_BINARY_DIR}/Extern/CGAL5.1/build)

set(CLOUD_CGAL_INCLUDE_DIR ${CLOUD_CGAL_SOURCE_DIR}/include)

set(CLOUD_BOOST_ROOT ${CLOUD_SOURCE_DIR}/Extern/boost_1_74_0)

if(UNIX)
    message("UNIX")
elseif(WIN32)
    ExternalProject_Add(
        CLOUD_CGAL
        PREFIX ${CLOUD_CGAL_BINARY_DIR}
        SOURCE_DIR ${CLOUD_CGAL_SOURCE_DIR}
        BINARY_DIR ${CLOUD_CGAL_BINARY_DIR}

        CMAKE_ARGS -DWITH_CGAL_Qt5=OFF -DCGAL_DIR=${CLOUD_CGAL_SOURCE_DIR} -DBOOST_ROOT=${CLOUD_BOOST_ROOT}
                -DCMAKE_INSTALL_PREFIX=${CLOUD_CGAL_BINARY_DIR}/install
    )
    set_target_properties(CLOUD_CGAL PROPERTIES FOLDER "${FOLDER_PREFIX}ExternalProjectTargets")

    set(CLOUD_gmp_INCLUDE_DIR ${CLOUD_CGAL_SOURCE_DIR}/auxiliary/gmp/include)
    set(CLOUD_mpfr_INCLUDE_DIR ${CLOUD_CGAL_SOURCE_DIR}/auxiliary/gmp/include)
    set(CLOUD_gmp_LIB_DIR ${CLOUD_CGAL_SOURCE_DIR}/auxiliary/gmp/lib)
    set(CLOUD_mpfr_LIB_DIR ${CLOUD_CGAL_SOURCE_DIR}/auxiliary/gmp/lib)

    add_library(CLOUD_GMP INTERFACE)

    add_library(libgmp UNKNOWN IMPORTED)
    set_target_properties(libgmp PROPERTIES IMPORTED_LOCATION "${CLOUD_gmp_LIB_DIR}/libgmp-10.lib")

    add_library(libmpfr UNKNOWN IMPORTED)
    set_target_properties(libmpfr PROPERTIES IMPORTED_LOCATION "${CLOUD_mpfr_LIB_DIR}/libmpfr-4.lib")

    target_link_libraries(CLOUD_GMP 
        INTERFACE libgmp
        INTERFACE libmpfr
    )

    set(cloud_gmp_shared_files_list "")
    list(APPEND cloud_gmp_shared_files_list "${CLOUD_gmp_LIB_DIR}/libgmp-10.dll")
    list(APPEND cloud_gmp_shared_files_list "${CLOUD_gmp_LIB_DIR}/libmpfr-4.dll")
else()
    message(FATAL_ERROR "Could not set install folders for this platform!")
endif()



# import netCDF-C
if(WIN32)
    set(CLOUD_netCDF_ROOT ${CLOUD_SOURCE_DIR}/Extern/netCDF-4.7.4)
    include_directories(
        ${CLOUD_netCDF_ROOT}/include
    )

    file(GLOB CLOUD_netCDF_LIBS ${CLOUD_netCDF_ROOT}/lib/*.lib)
    file(GLOB CLOUD_netCDF_DLLS ${CLOUD_netCDF_ROOT}/bin/*.dll)
    # message("${CLOUD_netCDF_LIBS}")
endif()



# import osg
option(USE_PREBUILT_OSG "option for OpenSceneGraph Windows Binaries" OFF)
option(CLOUD_WRF "option for wrf" ON)

if(USE_PREBUILT_OSG AND CLOUD_WRF)
    if(WIN32)
        set(CLOUD_OSG_ROOT ${CLOUD_SOURCE_DIR}/Extern/OpenSceneGraph-3.6.4-VC2017-64-Debug)
        include_directories(
            ${CLOUD_OSG_ROOT}/include
        )

        file(GLOB CLOUD_OSG_LIBS ${CLOUD_OSG_ROOT}/lib/*.lib)
        file(GLOB CLOUD_OSG_DLLS ${CLOUD_OSG_ROOT}/bin/*.dll)
    endif()
elseif(NOT USE_PREBUILT_OSG AND CLOUD_WRF)
    if(WIN32)
        include(${CLOUD_SOURCE_DIR}/OpenSceneGraph.cmake)
    endif()
endif()



# import OpenMesh part
set(CLOUD_OpenMesh_SOURCE_DIR ${CLOUD_SOURCE_DIR}/Extern/OpenMesh-8.1)
set(CLOUD_OpenMesh_BINARY_DIR ${CLOUD_BINARY_DIR}/Extern/openMesh-8.1/build)

set(CLOUD_OpenMesh_INCLUDE_DIR ${CLOUD_OpenMesh_SOURCE_DIR}/install-include)
set(CLOUD_OpenMesh_LIB_DIR ${CLOUD_OpenMesh_BINARY_DIR}/install/lib)

if(UNIX)
    message("UNIX")
elseif(WIN32)
    add_library(CLOUD_OpenMesh ${CLOUD_SOURCE_DIR}/Extern/openmesh_interface.cpp)
    set_target_properties(CLOUD_OpenMesh PROPERTIES FOLDER "${FOLDER_PREFIX}ExternalProjectTargets")

    add_custom_command(
        TARGET CLOUD_OpenMesh
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -S ${CLOUD_OpenMesh_SOURCE_DIR} -B ${CLOUD_OpenMesh_BINARY_DIR} -DBUILD_APPS=OFF -DOPENMESH_BUILD_SHARED=OFF -DOPENMESH_DOCS=OFF
                    -DOPENMESH_BUILD_UNIT_TESTS=OFF -A x64
        COMMAND ${CMAKE_COMMAND} --build ${CLOUD_OpenMesh_BINARY_DIR} --config $<$<CONFIG:>:Undefined>$<$<NOT:$<CONFIG:>>:$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} --install ${CLOUD_OpenMesh_BINARY_DIR} --prefix ${CLOUD_OpenMesh_BINARY_DIR}/install
                                 --config $<$<CONFIG:>:Undefined>$<$<NOT:$<CONFIG:>>:$<CONFIG>>
    )

    set(cloud_openmesh_libs
        OpenMeshCore
        OpenMeshTools
    )

    set(cloud_openmesh_link_list "")

    foreach(lib ${cloud_openmesh_libs})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES 
            IMPORTED_IMPLIB_DEBUG "${CLOUD_OpenMesh_LIB_DIR}/${lib}d.lib"
        )
        set_target_properties(${lib} PROPERTIES 
            IMPORTED_IMPLIB_RELEASE "${CLOUD_OpenMesh_LIB_DIR}/${lib}.lib"
        )
        list(APPEND cloud_openmesh_link_list ${lib})
    endforeach()

    target_link_libraries(CLOUD_OpenMesh
        ${cloud_openmesh_libs}
    )
endif()



## physika_cloud part
set(CLOUD_PYTHON_SOURCE_PATH ${CLOUD_SOURCE_DIR}/Python)
set(PYBIND11_DIR ${CLOUD_SOURCE_DIR}/Extern/pybind11)

set(CLOUD_Eigen_INCLUDE_DIR ${CLOUD_SOURCE_DIR}/Extern/Eigen-3.3.7)


include_directories(
    ${CLOUD_OpenCV_INCLUDE_DIR}

    ${CLOUD_Eigen_INCLUDE_DIR}

    ${CLOUD_OpenMesh_INCLUDE_DIR}
    
    ${CLOUD_CGAL_INCLUDE_DIR}
    ${CLOUD_BOOST_ROOT}
    ${CLOUD_gmp_INCLUDE_DIR}
    ${CLOUD_mpfr_INCLUDE_DIR}
)


add_library(COPY_DEPENDENCE ${CLOUD_SOURCE_DIR}/Extern/copy_file_dependence.cpp)
set_target_properties(COPY_DEPENDENCE PROPERTIES FOLDER "${FOLDER_PREFIX}ExternalProjectTargets")

option(CLOUD_PYTHON "option for python interface" ON)
if(CLOUD_PYTHON)
    add_custom_command(
        TARGET COPY_DEPENDENCE
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory 
                    ${CLOUD_SOURCE_DIR}/Python/
                    ${CLOUD_BINARY_DIR}/Python/
        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    ${CLOUD_SOURCE_DIR}/Cplusplus/SatImage/Typhoon/ground_temp_IR1.dat 
                    ${CLOUD_BINARY_DIR}/Python/Typhoon/xml
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    ${CLOUD_SOURCE_DIR}/Cplusplus/SatImage/Typhoon/ground_temp_IR2.dat 
                    ${CLOUD_BINARY_DIR}/Python/Typhoon/xml
    )
    if(NOT (${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR}) )
        if(NOT PhysIKA_Python_Binding)
            add_subdirectory(${CLOUD_SOURCE_DIR}/Extern)
        endif()
    else()
        add_subdirectory(${CLOUD_SOURCE_DIR}/Extern)
    endif()
    

endif()


add_subdirectory(${CLOUD_SOURCE_DIR}/Cplusplus)

add_subdirectory(${CLOUD_SOURCE_DIR}/Examples)
