#pragma once

#include <array>

// return the features of the model,
//[0]: is manifold in topology,
//[1]: is manifold in geometry
//[2]: is volume positive
std::array<bool, 3> manifold_checking(const char* obj_path);
