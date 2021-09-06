/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: extract surface.
 * @version    : 1.0
 */
#pragma once

#include <Eigen/Core>
#include <string>

/**
 * @brief extract the surface data
 * 
 * @tparam T 
 * @param nods 
 * @param cells 
 * @param surface 
 * @param type 
 * @return int 
 */
template <typename T>
int extract_surface(const Eigen::Matrix<T, -1, -1>& nods, const Eigen::MatrixXi& cells, Eigen::MatrixXi& surface, const std::string& type = "tet");
