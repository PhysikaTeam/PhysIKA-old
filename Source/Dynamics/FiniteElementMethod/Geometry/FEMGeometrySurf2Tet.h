#pragma once

#include <string>

/**
 * @brief Get the tetrahedron format file form surface file
 * 
 * @param surf_file 
 * @param tet_file 
 * @param num_span 
 * @return int 
 */
int surf2tet(const std::string surf_file, const std::string tet_file, const size_t num_span = 9);
