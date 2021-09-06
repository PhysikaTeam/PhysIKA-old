#pragma once

#include <Eigen/Core>

/**
 * @brief Determine whether the mesh is watertight
 * 
 * @param path 
 * @return true 
 * @return false 
 */
bool is_mesh_watertight(const char* const path);

/**
 * @brief Vert manifold
 * 
 * @param path 
 * @return true 
 * @return false 
 */
bool vert_manifold(const char* const path);

/**
 * @brief Determine whether it is the normal outside
 * 
 * @param path 
 * @return true 
 * @return false 
 */
bool is_normal_outside(const char* const path);

/**
 * @brief Get the triangle normal axis direction object
 * 
 * @param face 
 * @param axis 
 * @return int 
 */
int get_triangle_normal_axis_direction(const Eigen::Matrix3d& face, size_t axis);

/**
 * @brief Determine whether the volume of mesh is negative
 * 
 * @param path 
 * @return true 
 * @return false 
 */
bool is_mesh_volume_negative(const char* const path);
