# Search packages for host system instead of packages for target system
# in case of cross compilation thess macro should be defined by toolchain file
if(NOT COMMAND find_host_package)
  macro(find_host_package)
    find_package(${ARGN})
  endmacro()
endif()
if(NOT COMMAND find_host_program)
  macro(find_host_program)
    find_program(${ARGN})
  endmacro()
endif()

# assert macro
# Note: it doesn't support lists in arguments
# Usage samples:
#   ocv_assert(MyLib_FOUND)
#   ocv_assert(DEFINED MyLib_INCLUDE_DIRS)
macro(ocv_assert)
  if(NOT (${ARGN}))
    string(REPLACE ";" " " __assert_msg "${ARGN}")
    message(AUTHOR_WARNING "Assertion failed: ${__assert_msg}")
  endif()
endmacro()

macro(ocv_check_environment_variables)
  foreach(_var ${ARGN})
    if(NOT DEFINED ${_var} AND DEFINED ENV{${_var}})
      set(__value "$ENV{${_var}}")
      file(TO_CMAKE_PATH "${__value}" __value) # Assume that we receive paths
      set(${_var} "${__value}")
      message(STATUS "Update variable ${_var} from environment: ${${_var}}")
    endif()
  endforeach()
endmacro()

# adds include directories in such way that directories from the OpenCV source tree go first
function(ocv_include_directories)
  set(__add_before "")
  foreach(dir ${ARGN})
    get_filename_component(__abs_dir "${dir}" ABSOLUTE)
    if("${__abs_dir}" MATCHES "^${OpenCV_SOURCE_DIR}" OR "${__abs_dir}" MATCHES "^${OpenCV_BINARY_DIR}")
      list(APPEND __add_before "${dir}")
    else()
      include_directories(AFTER SYSTEM "${dir}")
    endif()
  endforeach()
  include_directories(BEFORE ${__add_before})
endfunction()

# clears all passed variables
macro(ocv_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var} CACHE)
  endforeach()
endmacro()

set(OCV_COMPILER_FAIL_REGEX
    "command line option .* is valid for .* but not for C\\+\\+" # GNU
    "command line option .* is valid for .* but not for C" # GNU
    "unrecognized .*option"                     # GNU
    "unknown .*option"                          # Clang
    "ignoring unknown option"                   # MSVC
    "warning D9002"                             # MSVC, any lang
    "option .*not supported"                    # Intel
    "[Uu]nknown option"                         # HP
    "[Ww]arning: [Oo]ption"                     # SunPro
    "command option .* is not recognized"       # XL
    "not supported in this configuration; ignored"       # AIX
    "File with unknown suffix passed to linker" # PGI
    "WARNING: unknown flag:"                    # Open64
  )

MACRO(ocv_check_compiler_flag LANG FLAG RESULT)
  if(NOT DEFINED ${RESULT})
    if("_${LANG}_" MATCHES "_CXX_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx")
      if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main() { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main() { return 0; }\n")
      endif()
    elseif("_${LANG}_" MATCHES "_C_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.c")
      if("${CMAKE_C_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_C_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main(void) { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main(void) { return 0; }\n")
      endif()
    elseif("_${LANG}_" MATCHES "_OBJCXX_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.mm")
      if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main() { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main() { return 0; }\n")
      endif()
    else()
      unset(_fname)
    endif()
    if(_fname)
      MESSAGE(STATUS "Performing Test ${RESULT}")
      TRY_COMPILE(${RESULT}
        "${CMAKE_BINARY_DIR}"
        "${_fname}"
        COMPILE_DEFINITIONS "${FLAG}"
        OUTPUT_VARIABLE OUTPUT)

      FOREACH(_regex ${OCV_COMPILER_FAIL_REGEX})
        IF("${OUTPUT}" MATCHES "${_regex}")
          SET(${RESULT} 0)
          break()
        ENDIF()
      ENDFOREACH()

      IF(${RESULT})
        SET(${RESULT} 1 CACHE INTERNAL "Test ${RESULT}")
        MESSAGE(STATUS "Performing Test ${RESULT} - Success")
      ELSE(${RESULT})
        MESSAGE(STATUS "Performing Test ${RESULT} - Failed")
        SET(${RESULT} "" CACHE INTERNAL "Test ${RESULT}")
      ENDIF(${RESULT})
    else()
      SET(${RESULT} 0)
    endif()
  endif()
ENDMACRO()

macro(ocv_check_flag_support lang flag varname)
  if("_${lang}_" MATCHES "_CXX_")
    set(_lang CXX)
  elseif("_${lang}_" MATCHES "_C_")
    set(_lang C)
  elseif("_${lang}_" MATCHES "_OBJCXX_")
    set(_lang OBJCXX)
  else()
    set(_lang ${lang})
  endif()

  string(TOUPPER "${flag}" ${varname})
  string(REGEX REPLACE "^(/|-)" "HAVE_${_lang}_" ${varname} "${${varname}}")
  string(REGEX REPLACE " -|-|=| |\\." "_" ${varname} "${${varname}}")

  ocv_check_compiler_flag("${_lang}" "${ARGN} ${flag}" ${${varname}})
endmacro()

# turns off warnings
macro(ocv_warnings_disable)
  if(NOT ENABLE_NOISY_WARNINGS)
    set(_flag_vars "")
    set(_msvc_warnings "")
    set(_gxx_warnings "")
    foreach(arg ${ARGN})
      if(arg MATCHES "^CMAKE_")
        list(APPEND _flag_vars ${arg})
      elseif(arg MATCHES "^/wd")
        list(APPEND _msvc_warnings ${arg})
      elseif(arg MATCHES "^-W")
        list(APPEND _gxx_warnings ${arg})
      endif()
    endforeach()
    if(MSVC AND _msvc_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_msvc_warnings})
          set(${var} "${${var}} ${warning}")
        endforeach()
      endforeach()
    elseif((CMAKE_COMPILER_IS_GNUCXX OR (UNIX AND CV_ICC)) AND _gxx_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_gxx_warnings})
          if(NOT warning MATCHES "^-Wno-")
            string(REPLACE "${warning}" "" ${var} "${${var}}")
            string(REPLACE "-W" "-Wno-" warning "${warning}")
          endif()
          ocv_check_flag_support(${var} "${warning}" _varname)
          if(${_varname})
            set(${var} "${${var}} ${warning}")
          endif()
        endforeach()
      endforeach()
    endif()
    unset(_flag_vars)
    unset(_msvc_warnings)
    unset(_gxx_warnings)
  endif(NOT ENABLE_NOISY_WARNINGS)
endmacro()

# Provides an option that the user can optionally select.
# Can accept condition to control when option is available for user.
# Usage:
#   option(<option_variable> "help string describing the option" <initial value or boolean expression> [IF <condition>])
macro(OCV_OPTION variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if(__condition STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if(__value MATCHES ";")
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    else()
      option(${variable} "${description}" ${__value})
    endif()
  else()
    unset(${variable} CACHE)
  endif()
  unset(__condition)
  unset(__value)
endmacro()


# Macros that checks if module have been installed.
# After it adds module to build and define
# constants passed as second arg
macro(CHECK_MODULE module_name define)
  set(${define} 0)
  if(PKG_CONFIG_FOUND)
    set(ALIAS               ALIASOF_${module_name})
    set(ALIAS_FOUND                 ${ALIAS}_FOUND)
    set(ALIAS_INCLUDE_DIRS   ${ALIAS}_INCLUDE_DIRS)
    set(ALIAS_LIBRARY_DIRS   ${ALIAS}_LIBRARY_DIRS)
    set(ALIAS_LIBRARIES         ${ALIAS}_LIBRARIES)

    PKG_CHECK_MODULES(${ALIAS} ${module_name})

    if(${ALIAS_FOUND})
      set(${define} 1)
      foreach(P "${ALIAS_INCLUDE_DIRS}")
        if(${P})
          list(APPEND HIGHGUI_INCLUDE_DIRS ${${P}})
        endif()
      endforeach()

      foreach(P "${ALIAS_LIBRARY_DIRS}")
        if(${P})
          list(APPEND HIGHGUI_LIBRARY_DIRS ${${P}})
        endif()
      endforeach()

      list(APPEND HIGHGUI_LIBRARIES ${${ALIAS_LIBRARIES}})
    endif()
  endif()
endmacro()


set(OPENCV_BUILD_INFO_FILE "${OpenCV_BINARY_DIR}/version_string.tmp")
file(REMOVE "${OPENCV_BUILD_INFO_FILE}")
function(ocv_output_status msg)
  message(STATUS "${msg}")
  string(REPLACE "\\" "\\\\" msg "${msg}")
  string(REPLACE "\"" "\\\"" msg "${msg}")
  file(APPEND "${OPENCV_BUILD_INFO_FILE}" "\"${msg}\\n\"\n")
endfunction()

macro(ocv_finalize_status)
  if(NOT OPENCV_SKIP_STATUS_FINALIZATION)
    if(TARGET opencv_core)
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_BUILD_INFO_FILE}" "${opencv_core_BINARY_DIR}/version_string.inc" OUTPUT_QUIET)
    endif()
  endif()
endmacro()


# Status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   status(<text>)
#   status(<heading> <value1> [<value2> ...])
#   status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    endif()
  endforeach()

  if(DEFINED status_cond)
    set(status_placeholder_length 32)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif(DEFINED status_then OR DEFINED status_else)
      ocv_output_status("${text}")
      set(status_text "${status_placeholder}")
    else()
      set(status_text "${text}")
    endif()

    if(DEFINED status_then OR DEFINED status_else)
      if(${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        ocv_output_status("${status_text} ${status_then}")
      else()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        ocv_output_status("${status_text} ${status_else}")
      endif()
    else()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      ocv_output_status("${status_text} ${status_cond}")
    endif()
  else()
    ocv_output_status("${text}")
  endif()
endfunction()


# splits cmake libraries list of format "general;item1;debug;item2;release;item3" to two lists
macro(ocv_split_libs_list lst lstdbg lstopt)
  set(${lstdbg} "")
  set(${lstopt} "")
  set(perv_keyword "")
  foreach(word ${${lst}})
    if(word STREQUAL "debug" OR word STREQUAL "optimized")
      set(perv_keyword ${word})
    elseif(word STREQUAL "general")
      set(perv_keyword "")
    elseif(perv_keyword STREQUAL "debug")
      list(APPEND ${lstdbg} "${word}")
      set(perv_keyword "")
    elseif(perv_keyword STREQUAL "optimized")
      list(APPEND ${lstopt} "${word}")
      set(perv_keyword "")
    else()
      list(APPEND ${lstdbg} "${word}")
      list(APPEND ${lstopt} "${word}")
      set(perv_keyword "")
    endif()
  endforeach()
endmacro()


# remove all matching elements from the list
macro(ocv_list_filterout lst regex)
  foreach(item ${${lst}})
    if(item MATCHES "${regex}")
      list(REMOVE_ITEM ${lst} "${item}")
    endif()
  endforeach()
endmacro()


# stable & safe duplicates removal macro
macro(ocv_list_unique __lst)
  if(${__lst})
    list(REMOVE_DUPLICATES ${__lst})
  endif()
endmacro()


# safe list reversal macro
macro(ocv_list_reverse __lst)
  if(${__lst})
    list(REVERSE ${__lst})
  endif()
endmacro()


# safe list sorting macro
macro(ocv_list_sort __lst)
  if(${__lst})
    list(SORT ${__lst})
  endif()
endmacro()


# add prefix to each item in the list
macro(ocv_list_add_prefix LST PREFIX)
  set(__tmp "")
  foreach(item ${${LST}})
    list(APPEND __tmp "${PREFIX}${item}")
  endforeach()
  set(${LST} ${__tmp})
  unset(__tmp)
endmacro()


# add suffix to each item in the list
macro(ocv_list_add_suffix LST SUFFIX)
  set(__tmp "")
  foreach(item ${${LST}})
    list(APPEND __tmp "${item}${SUFFIX}")
  endforeach()
  set(${LST} ${__tmp})
  unset(__tmp)
endmacro()


# gets and removes the first element from list
macro(ocv_list_pop_front LST VAR)
  if(${LST})
    list(GET ${LST} 0 ${VAR})
    list(REMOVE_AT ${LST} 0)
  else()
    set(${VAR} "")
  endif()
endmacro()


# simple regex escaping routine (does not cover all cases!!!)
macro(ocv_regex_escape var regex)
  string(REGEX REPLACE "([+.*^$])" "\\\\1" ${var} "${regex}")
endmacro()


# convert list of paths to full paths
macro(ocv_convert_to_full_paths VAR)
  if(${VAR})
    set(__tmp "")
    foreach(path ${${VAR}})
      get_filename_component(${VAR} "${path}" ABSOLUTE)
      list(APPEND __tmp "${${VAR}}")
    endforeach()
    set(${VAR} ${__tmp})
    unset(__tmp)
  endif()
endmacro()


# convert list of paths to libraries names without lib prefix
function(ocv_convert_to_lib_name var)
  set(tmp "")
  foreach(path ${ARGN})
    get_filename_component(tmp_name "${path}" NAME_WE)
    string(REGEX REPLACE "^lib" "" tmp_name "${tmp_name}")
    list(APPEND tmp "${tmp_name}")
  endforeach()
  set(${var} ${tmp} PARENT_SCOPE)
endfunction()

# create imported targets for a list of external libraries
function(ocv_create_imported_targets var)
  set(target_list "")

  foreach(library ${ARGN})
    ocv_convert_to_lib_name(libname "${library}")
    add_library("opencv_dep_${libname}" UNKNOWN IMPORTED)
    set_target_properties("opencv_dep_${libname}" PROPERTIES IMPORTED_LOCATION "${library}")
    list(APPEND target_list "opencv_dep_${libname}")
  endforeach()

  set("${var}" "${target_list}" PARENT_SCOPE)
endfunction()

# add install command
function(ocv_install_target)
  install(TARGETS ${ARGN})

  set(isPackage 0)
  unset(__package)
  unset(__target)
  foreach(e ${ARGN})
    if(NOT DEFINED __target)
      set(__target "${e}")
    endif()
    if(isPackage EQUAL 1)
      set(__package "${e}")
      break()
    endif()
    if(e STREQUAL "EXPORT")
      set(isPackage 1)
    endif()
  endforeach()

  if(DEFINED __package)
    list(APPEND ${__package}_TARGETS ${__target})
    set(${__package}_TARGETS "${${__package}_TARGETS}" CACHE INTERNAL "List of ${__package} targets")
  endif()

  if(INSTALL_CREATE_DISTRIB)
    if(MSVC AND NOT BUILD_SHARED_LIBS)
      set(__target "${ARGV0}")

      set(isArchive 0)
      set(isDst 0)
      unset(__dst)
      foreach(e ${ARGN})
        if(isDst EQUAL 1)
          set(__dst "${e}")
          break()
        endif()
        if(isArchive EQUAL 1 AND e STREQUAL "DESTINATION")
          set(isDst 1)
        endif()
        if(e STREQUAL "ARCHIVE")
          set(isArchive 1)
        else()
          set(isArchive 0)
        endif()
      endforeach()

#      message(STATUS "Process ${__target} dst=${__dst}...")
      if(DEFINED __dst)
        get_target_property(fname ${__target} LOCATION_DEBUG)
        if(fname MATCHES "\\.lib$")
          string(REGEX REPLACE "\\.lib$" ".pdb" fname "${fname}")
          install(FILES ${fname} DESTINATION ${__dst} CONFIGURATIONS Debug)
        endif()

        get_target_property(fname ${__target} LOCATION_RELEASE)
        if(fname MATCHES "\\.lib$")
          string(REGEX REPLACE "\\.lib$" ".pdb" fname "${fname}")
          install(FILES ${fname} DESTINATION ${__dst} CONFIGURATIONS Release)
        endif()
      endif()
    endif()
  endif()
endfunction()


# read set of version defines from the header file
macro(ocv_parse_header FILENAME FILE_VAR)
  set(vars_regex "")
  set(__parnet_scope OFF)
  set(__add_cache OFF)
  foreach(name ${ARGN})
    if("${name}" STREQUAL "PARENT_SCOPE")
      set(__parnet_scope ON)
    elseif("${name}" STREQUAL "CACHE")
      set(__add_cache ON)
    elseif(vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else()
      set(vars_regex "${name}")
    endif()
  endforeach()
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${FILE_VAR} REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+" )
  else()
    unset(${FILE_VAR})
  endif()
  foreach(name ${ARGN})
    if(NOT "${name}" STREQUAL "PARENT_SCOPE" AND NOT "${name}" STREQUAL "CACHE")
      if(${FILE_VAR})
        if(${FILE_VAR} MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
          string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" ${name} "${${FILE_VAR}}")
        else()
          set(${name} "")
        endif()
        if(__add_cache)
          set(${name} ${${name}} CACHE INTERNAL "${name} parsed from ${FILENAME}" FORCE)
        elseif(__parnet_scope)
          set(${name} "${${name}}" PARENT_SCOPE)
        endif()
      else()
        unset(${name} CACHE)
      endif()
    endif()
  endforeach()
endmacro()

# read single version define from the header file
macro(ocv_parse_header2 LIBNAME HDR_PATH VARNAME)
  ocv_clear_vars(${LIBNAME}_VERSION_MAJOR
                 ${LIBNAME}_VERSION_MAJOR
                 ${LIBNAME}_VERSION_MINOR
                 ${LIBNAME}_VERSION_PATCH
                 ${LIBNAME}_VERSION_TWEAK
                 ${LIBNAME}_VERSION_STRING)
  set(${LIBNAME}_H "")
  if(EXISTS "${HDR_PATH}")
    file(STRINGS "${HDR_PATH}" ${LIBNAME}_H REGEX "^#define[ \t]+${VARNAME}[ \t]+\"[^\"]*\".*$" LIMIT_COUNT 1)
  endif()

  if(${LIBNAME}_H)
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${${LIBNAME}_H}")
    set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN})
    set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN})
    set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN})
    set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}")

    # append a TWEAK version if it exists:
    set(${LIBNAME}_VERSION_TWEAK "")
    if("${${LIBNAME}_H}" MATCHES "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+).*$")
      set(${LIBNAME}_VERSION_TWEAK "${CMAKE_MATCH_1}" ${ARGN})
    endif()
    if(${LIBNAME}_VERSION_TWEAK)
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}.${${LIBNAME}_VERSION_TWEAK}" ${ARGN})
    else()
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}" ${ARGN})
    endif()
  endif()
endmacro()

# read single version info from the pkg file
macro(ocv_parse_pkg LIBNAME PKG_PATH SCOPE)
  if(EXISTS "${PKG_PATH}/${LIBNAME}.pc")
    file(STRINGS "${PKG_PATH}/${LIBNAME}.pc" line_to_parse REGEX "^Version:[ \t]+[0-9.]*.*$" LIMIT_COUNT 1)
    STRING(REGEX REPLACE ".*Version: ([^ ]+).*" "\\1" ALIASOF_${LIBNAME}_VERSION "${line_to_parse}" )
  endif()
endmacro()

################################################################################################
# short command to setup source group
function(ocv_source_group group)
  cmake_parse_arguments(OCV_SOURCE_GROUP "" "" "GLOB" ${ARGN})
  file(GLOB srcs ${OCV_SOURCE_GROUP_GLOB})
  source_group(${group} FILES ${srcs})
endfunction()

# build the list of simple dependencies, that links via "-l"
#  _all_libs - name of variable with input list
#  _simple - name of variable with output list of simple libs
#  _other - name of variable with _all_libs - _simple
macro(ocv_extract_simple_libs _all_libs _simple _other)
  set(${_simple} "")
  set(${_other} "")
  foreach(_l ${${_all_libs}})
    if(TARGET ${_l})
        list(APPEND ${_other} ${_l})
    elseif(EXISTS "${_l}")
        list(APPEND ${_other} ${_l})
    else()
        list(APPEND ${_simple} ${_l})
    endif()
  endforeach()
endmacro()
