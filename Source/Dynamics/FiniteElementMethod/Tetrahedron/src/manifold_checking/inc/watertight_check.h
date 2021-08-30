#ifndef WATERTIGHT_CHECK_JJ_H
#define WATERTIGHT_CHECK_JJ_H

#include <Eigen/Core>

bool is_mesh_watertight(const char* const path);

bool vert_manifold(const char* const path);

bool is_normal_outside(const char* const path);

int get_triangle_normal_axis_direction(const Eigen::Matrix3d& face, size_t axis);

bool is_mesh_volume_negative(const char* const path);

#endif  // WATERTIGHT_CHECK_JJ_H
