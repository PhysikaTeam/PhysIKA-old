#pragma once

/**
 * @brief The initialize function
 * 
 */
void   exactinit();

/**
 * @brief The exact 2D orientation
 * 
 * @param pa 
 * @param pb 
 * @param pc 
 * @return double 
 */
double orient2d(double const* pa, double const* pb, double const* pc);

/**
 * @brief The exact 3D orientation
 * 
 * @param pa 
 * @param pb 
 * @param pc 
 * @param pd 
 * @return double 
 */
double orient3d(double const* pa, double const* pb, double const* pc, double const* pd);
