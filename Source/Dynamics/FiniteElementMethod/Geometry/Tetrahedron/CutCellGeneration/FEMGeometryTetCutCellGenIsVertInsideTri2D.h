#pragma once

#include "Predicates.h"

#include <cstddef>
#include <vector>
#include <Eigen/Core>

/**
 * @brief Determine whether the point is inside the triangle.
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param p 
 * @return int 
 */
int is_vert_inside_triangle_2d(const double* const a, const double* const b, const double* const c, const double* const p);

/**
 * @brief Determine whether the point is inside the triangle
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param p 
 * @param v1_e 
 * @param v2_e 
 * @return int 
 */
int is_vert_inside_triangle_2d(const double* const a, const double* const b, const double* const c, const double* const p, size_t& v1_e, size_t& v2_e);

/**
 * @brief Determine whether the point is above the triangle
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param p 
 * @return int 
 */
int is_vert_above_triangle(const double* const a, const double* const b, const double* const c, const double* const p);

/**
 * @brief Determine whether the eare of the triangle is positive.
 * 
 * @param axis 
 * @param tri_v 
 * @return int 
 */
int is_triangle_area_positive(
    size_t                              axis,
    const std::vector<Eigen::Vector3d>& tri_v);

/**
 * @brief Determine whether the eare of the triangle is positive.
 * 
 * @param pa 
 * @param pb 
 * @param pc 
 * @return int 
 */
int is_triangle_area_positive(double const* pa, double const* pb, double const* pc);
