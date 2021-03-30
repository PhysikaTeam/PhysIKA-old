# Require dot, treat the other components as optional
find_package(Doxygen)

option(BUILD_DOCUMENTATION
  "Create and install the HTML based API documentation (requires Doxygen)"
  ${DOXYGEN_FOUND})

if(NOT DOXYGEN_FOUND)
  message(FATAL_ERROR "Doxygen is needed to build the documentation.")
endif()

message(STATUS "DOXYGEN_VERSION     : ${DOXYGEN_VERSION}")
set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${doxyfile_in} ${doxyfile} @ONLY)

# NOT TO COMPILE DOC by default
add_custom_target(doc# ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
