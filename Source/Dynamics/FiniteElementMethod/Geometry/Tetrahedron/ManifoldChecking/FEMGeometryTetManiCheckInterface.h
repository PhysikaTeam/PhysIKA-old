#pragma once

#include <array>

/**
 * @brief Return the features of the model
 * 
 * @param obj_path 
 * @return std::array<bool, 3>
 *   [0]: is manifold in topology,
 *   [1]: is manifold in geometry
 *   [2]: is volume positive 
 */
std::array<bool, 3> manifold_checking(const char* obj_path);
