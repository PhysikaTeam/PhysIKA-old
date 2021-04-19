# taken from http://public.kitware.com/Bug/view.php?id=1260 and slightly adjusted

# - Try to find precompiled headers support for GCC 3.4 and 4.x
# Once done this will define:
#
# Variable:
#   PCHSupport_FOUND
#
# Macro:
#   ADD_PRECOMPILED_HEADER  _targetName _input  _dowarn
#   ADD_PRECOMPILED_HEADER_TO_TARGET _targetName _input _pch_output_to_use _dowarn
#   ADD_NATIVE_PRECOMPILED_HEADER _targetName _input _dowarn
#   GET_NATIVE_PRECOMPILED_HEADER _targetName _input

IF(CMAKE_COMPILER_IS_GNUCXX)

    EXEC_PROGRAM(
        ${CMAKE_CXX_COMPILER}
        ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE gcc_compiler_version)
    #MESSAGE("GCC Version: ${gcc_compiler_version}")
    IF(gcc_compiler_version VERSION_GREATER "4.2.-1")
        SET(PCHSupport_FOUND TRUE)
    ENDIF()

    SET(_PCH_include_prefix "-I")
    SET(_PCH_isystem_prefix "-isystem")
    SET(_PCH_define_prefix "-D")

ELSEIF(CMAKE_GENERATOR MATCHES "^Visual.*$")
    SET(PCHSupport_FOUND TRUE)
    SET(_PCH_include_prefix "/I")
    SET(_PCH_isystem_prefix "/I")
    SET(_PCH_define_prefix "/D")
ELSE()
    SET(PCHSupport_FOUND FALSE)
ENDIF()

MACRO(_PCH_GET_COMPILE_FLAGS _out_compile_flags)

    STRING(TOUPPER "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}" _flags_var_name)
    SET(${_out_compile_flags} ${${_flags_var_name}} )

    GET_TARGET_PROPERTY(_targetPIC ${_PCH_current_target} POSITION_INDEPENDENT_CODE)
    if (_targetPIC AND CMAKE_CXX_COMPILE_OPTIONS_PIC)
      LIST(APPEND ${_out_compile_flags} "${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
    elseif(CMAKE_COMPILER_IS_GNUCXX)
      GET_TARGET_PROPERTY(_targetType ${_PCH_current_target} TYPE)
      IF(${_targetType} STREQUAL SHARED_LIBRARY AND NOT WIN32)
        LIST(APPEND ${_out_compile_flags} "-fPIC")
      ENDIF()
    endif()

    IF(CMAKE_COMPILER_IS_GNUCXX)

        GET_PROPERTY(_definitions DIRECTORY PROPERTY COMPILE_DEFINITIONS)
        if(_definitions)
          foreach(_def ${_definitions})
            LIST(APPEND ${_out_compile_flags} "\"-D${_def}\"")
          endforeach()
        endif()
        GET_TARGET_PROPERTY(_target_definitions ${_PCH_current_target} COMPILE_DEFINITIONS)
        if(_target_definitions)
          foreach(_def ${_target_definitions})
            LIST(APPEND ${_out_compile_flags} "\"-D${_def}\"")
          endforeach()
        endif()

    ELSE()
        ## TODO ... ? or does it work out of the box
    ENDIF()

    GET_DIRECTORY_PROPERTY(DIRINC INCLUDE_DIRECTORIES )
    FOREACH(item ${DIRINC})
        if(item MATCHES "^${OpenCV_SOURCE_DIR}/modules/")
          LIST(APPEND ${_out_compile_flags} "${_PCH_include_prefix}\"${item}\"")
        else()
          LIST(APPEND ${_out_compile_flags} "${_PCH_isystem_prefix}\"${item}\"")
        endif()
    ENDFOREACH(item)

    get_target_property(DIRINC ${_PCH_current_target} INCLUDE_DIRECTORIES )
    FOREACH(item ${DIRINC})
        if(item MATCHES "^${OpenCV_SOURCE_DIR}/modules/")
          LIST(APPEND ${_out_compile_flags} "${_PCH_include_prefix}\"${item}\"")
        else()
          LIST(APPEND ${_out_compile_flags} "${_PCH_isystem_prefix}\"${item}\"")
        endif()
    ENDFOREACH(item)

    LIST(APPEND ${_out_compile_flags} ${CMAKE_CXX_FLAGS})

    SEPARATE_ARGUMENTS(${_out_compile_flags})

ENDMACRO(_PCH_GET_COMPILE_FLAGS)


MACRO(_PCH_WRITE_PCHDEP_CXX _targetName _include_file _dephelp)

    SET(${_dephelp} ${CMAKE_CURRENT_BINARY_DIR}/${_targetName}_pch_dephelp.cxx)
    IF(CMAKE_HOST_WIN32)
        ADD_CUSTOM_COMMAND(
          OUTPUT "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "#include \\\"${_include_file}\\\"" >  "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "int testfunction();"               >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "int testfunction()"                >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "{"                                 >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "    return 0;"                     >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "}"                                 >> "${${_dephelp}}"
          DEPENDS "${_include_file}"
          )
    else()
        ADD_CUSTOM_COMMAND(
          OUTPUT "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "\\#include \\\"${_include_file}\\\"" >  "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "int testfunction\\(\\)\\;"         >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "int testfunction\\(\\)"            >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "{"                                 >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "    \\return 0\\;"                 >> "${${_dephelp}}"
          COMMAND ${CMAKE_COMMAND} -E echo "}"                                 >> "${${_dephelp}}"
          DEPENDS "${_include_file}"
          )
    endif()

ENDMACRO(_PCH_WRITE_PCHDEP_CXX )

MACRO(_PCH_GET_COMPILE_COMMAND out_command _input _output)

    FILE(TO_NATIVE_PATH ${_input} _native_input)
    FILE(TO_NATIVE_PATH ${_output} _native_output)

    IF(CMAKE_COMPILER_IS_GNUCXX)
        IF(CMAKE_CXX_COMPILER_ARG1)
            # remove leading space in compiler argument
            STRING(REGEX REPLACE "^ +" "" pchsupport_compiler_cxx_arg1 ${CMAKE_CXX_COMPILER_ARG1})

            SET(${out_command}
              ${CMAKE_CXX_COMPILER} ${pchsupport_compiler_cxx_arg1} ${_compile_FLAGS} -x c++-header -o ${_output} ${_input}
              )
        ELSE(CMAKE_CXX_COMPILER_ARG1)
            SET(${out_command}
              ${CMAKE_CXX_COMPILER}  ${_compile_FLAGS} -x c++-header -o ${_output} ${_input}
              )
        ENDIF(CMAKE_CXX_COMPILER_ARG1)
    ELSE(CMAKE_COMPILER_IS_GNUCXX)

        SET(_dummy_str "#include <${_input}>")
        FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pch_dummy.cpp ${_dummy_str})

        SET(${out_command}
          ${CMAKE_CXX_COMPILER} ${_compile_FLAGS} /c /Fp${_native_output} /Yc${_native_input} pch_dummy.cpp
          )
        #/out:${_output}

    ENDIF(CMAKE_COMPILER_IS_GNUCXX)

ENDMACRO(_PCH_GET_COMPILE_COMMAND )


MACRO(_PCH_GET_TARGET_COMPILE_FLAGS _cflags  _header_name _pch_path _dowarn )

    FILE(TO_NATIVE_PATH ${_pch_path} _native_pch_path)

    IF(CMAKE_COMPILER_IS_GNUCXX)
        # for use with distcc and gcc >4.0.1 if preprocessed files are accessible
        # on all remote machines set
        # PCH_ADDITIONAL_COMPILER_FLAGS to -fpch-preprocess
        # if you want warnings for invalid header files (which is very inconvenient
        # if you have different versions of the headers for different build types
        # you may set _pch_dowarn
        IF (_dowarn)
            SET(${_cflags} "${PCH_ADDITIONAL_COMPILER_FLAGS} -Winvalid-pch " )
        ELSE (_dowarn)
            SET(${_cflags} "${PCH_ADDITIONAL_COMPILER_FLAGS} " )
        ENDIF (_dowarn)

    ELSE(CMAKE_COMPILER_IS_GNUCXX)

        set(${_cflags} "/Fp${_native_pch_path} /Yu${_header_name}" )

    ENDIF(CMAKE_COMPILER_IS_GNUCXX)

ENDMACRO(_PCH_GET_TARGET_COMPILE_FLAGS )


MACRO(GET_PRECOMPILED_HEADER_OUTPUT _targetName _input _output)

    GET_FILENAME_COMPONENT(_name ${_input} NAME)
    GET_FILENAME_COMPONENT(_path ${_input} PATH)
    SET(${_output} "${CMAKE_CURRENT_BINARY_DIR}/${_name}.gch/${_targetName}_${CMAKE_BUILD_TYPE}.gch")

ENDMACRO(GET_PRECOMPILED_HEADER_OUTPUT _targetName _input)


MACRO(ADD_PRECOMPILED_HEADER_TO_TARGET _targetName _input _pch_output_to_use )

    # to do: test whether compiler flags match between target  _targetName
    # and _pch_output_to_use
    GET_FILENAME_COMPONENT(_name ${_input} NAME)

    IF(ARGN STREQUAL "0")
        SET(_dowarn 0)
    ELSE()
        SET(_dowarn 1)
    ENDIF()

    _PCH_GET_TARGET_COMPILE_FLAGS(_target_cflags ${_name} ${_pch_output_to_use} ${_dowarn})
    #MESSAGE("Add flags ${_target_cflags} to ${_targetName} " )

    GET_TARGET_PROPERTY(_sources ${_targetName} SOURCES)
    FOREACH(src ${_sources})
      if(NOT "${src}" MATCHES "\\.mm$")
        get_source_file_property(_flags "${src}" COMPILE_FLAGS)
        if(_flags)
          set(_flags "${_flags} ${_target_cflags}")
        else()
          set(_flags "${_target_cflags}")
        endif()

        set_source_files_properties("${src}" PROPERTIES COMPILE_FLAGS "${_flags}")
      endif()
    ENDFOREACH()

    ADD_CUSTOM_TARGET(pch_Generate_${_targetName}
      DEPENDS ${_pch_output_to_use}
      )

    ADD_DEPENDENCIES(${_targetName} pch_Generate_${_targetName} )

ENDMACRO(ADD_PRECOMPILED_HEADER_TO_TARGET)

MACRO(ADD_PRECOMPILED_HEADER _targetName _input)

    SET(_PCH_current_target ${_targetName})

    IF(NOT CMAKE_BUILD_TYPE)
        MESSAGE(FATAL_ERROR
          "This is the ADD_PRECOMPILED_HEADER macro. "
          "You must set CMAKE_BUILD_TYPE!"
          )
    ENDIF()

    IF(ARGN STREQUAL "0")
        SET(_dowarn 0)
    ELSE()
        SET(_dowarn 1)
    ENDIF()

    GET_FILENAME_COMPONENT(_name ${_input} NAME)
    GET_FILENAME_COMPONENT(_path ${_input} PATH)
    GET_PRECOMPILED_HEADER_OUTPUT( ${_targetName} ${_input} _output)

    _PCH_WRITE_PCHDEP_CXX(${_targetName} "${_input}" _pch_dephelp_cxx)

    ADD_LIBRARY(${_targetName}_pch_dephelp STATIC "${_pch_dephelp_cxx}" "${_input}" )

    set_target_properties(${_targetName}_pch_dephelp PROPERTIES
      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
      ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
      )

    _PCH_GET_COMPILE_FLAGS(_compile_FLAGS)

    get_target_property(type ${_targetName} TYPE)
    if(type STREQUAL "SHARED_LIBRARY")
        get_target_property(__DEFINES ${_targetName} DEFINE_SYMBOL)
        if(NOT __DEFINES MATCHES __DEFINES-NOTFOUND)
            list(APPEND _compile_FLAGS "${_PCH_define_prefix}${__DEFINES}")
        endif()
    endif()

    get_target_property(DIRINC ${_targetName} INCLUDE_DIRECTORIES)
    set_target_properties(${_targetName}_pch_dephelp PROPERTIES INCLUDE_DIRECTORIES "${DIRINC}")

    #MESSAGE("_compile_FLAGS: ${_compile_FLAGS}")
    #message("COMMAND ${CMAKE_CXX_COMPILER}	${_compile_FLAGS} -x c++-header -o ${_output} ${_input}")

    ADD_CUSTOM_COMMAND(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_name}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_input}" "${CMAKE_CURRENT_BINARY_DIR}/${_name}" # ensure same directory! Required by gcc
      DEPENDS "${_input}"
      )

    #message("_command  ${_input} ${_output}")
    _PCH_GET_COMPILE_COMMAND(_command  ${CMAKE_CURRENT_BINARY_DIR}/${_name} ${_output} )

    GET_FILENAME_COMPONENT(_outdir ${_output} PATH)
    ADD_CUSTOM_COMMAND(
      OUTPUT "${_output}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_outdir}"
      COMMAND ${_command}
      DEPENDS "${_input}"
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_name}"
      DEPENDS ${_targetName}_pch_dephelp
      )

    ADD_PRECOMPILED_HEADER_TO_TARGET(${_targetName} ${_input}  ${_output} ${_dowarn})

ENDMACRO(ADD_PRECOMPILED_HEADER)


# Generates the use of precompiled in a target,
# without using depency targets (2 extra for each target)
# Using Visual, must also add ${_targetName}_pch to sources
# Not needed by Xcode

MACRO(GET_NATIVE_PRECOMPILED_HEADER _targetName _input)

    if(CMAKE_GENERATOR MATCHES "^Visual.*$")
        set(_dummy_str "#include \"${_input}\"\n")

        set(${_targetName}_pch ${CMAKE_CURRENT_BINARY_DIR}/${_targetName}_pch.cpp)
        if(EXISTS ${${_targetName}_pch})
            # Check if contents is the same, if not rewrite
            # todo
        else()
            FILE(WRITE ${${_targetName}_pch} ${_dummy_str})
        endif()
    endif()

ENDMACRO(GET_NATIVE_PRECOMPILED_HEADER)


MACRO(ADD_NATIVE_PRECOMPILED_HEADER _targetName _input)

    IF(ARGN STREQUAL "0")
        SET(_dowarn 0)
    ELSE()
        SET(_dowarn 1)
    ENDIF()

    if(CMAKE_GENERATOR MATCHES "^Visual.*$")

        # Auto include the precompile (useful for moc processing, since the use of
        # precompiled is specified at the target level
        # and I don't want to specifiy /F- for each moc/res/ui generated files (using Qt)

        GET_TARGET_PROPERTY(oldProps ${_targetName} COMPILE_FLAGS)
        if (oldProps MATCHES NOTFOUND)
            SET(oldProps "")
        endif()

        SET(newProperties "${oldProps} /Yu\"${_input}\" /FI\"${_input}\"")
        SET_TARGET_PROPERTIES(${_targetName} PROPERTIES COMPILE_FLAGS "${newProperties}")

        #also inlude ${oldProps} to have the same compile options
        SET_SOURCE_FILES_PROPERTIES(${${_targetName}_pch} PROPERTIES COMPILE_FLAGS "${oldProps} /Yc\"${_input}\"")

    elseif (CMAKE_GENERATOR MATCHES Xcode)

        # For Xcode, cmake needs my patch to process
        # GCC_PREFIX_HEADER and GCC_PRECOMPILE_PREFIX_HEADER as target properties

        # When buiding out of the tree, precompiled may not be located
        # Use full path instead.
        GET_FILENAME_COMPONENT(fullPath ${_input} ABSOLUTE)

        SET_TARGET_PROPERTIES(${_targetName} PROPERTIES XCODE_ATTRIBUTE_GCC_PREFIX_HEADER "${fullPath}")
        SET_TARGET_PROPERTIES(${_targetName} PROPERTIES XCODE_ATTRIBUTE_GCC_PRECOMPILE_PREFIX_HEADER "YES")

    else()

        #Fallback to the "old" precompiled suppport
        #ADD_PRECOMPILED_HEADER(${_targetName} ${_input} ${_dowarn})

    endif()

ENDMACRO(ADD_NATIVE_PRECOMPILED_HEADER)

macro(ocv_add_precompiled_header_to_target the_target pch_header)
  if(PCHSupport_FOUND AND ENABLE_PRECOMPILED_HEADERS AND EXISTS "${pch_header}")
    if(CMAKE_GENERATOR MATCHES "^Visual" OR CMAKE_GENERATOR MATCHES Xcode)
      add_native_precompiled_header(${the_target} ${pch_header})
    elseif(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
      add_precompiled_header(${the_target} ${pch_header})
    endif()
  endif()
endmacro()
