/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: extract surface.
 * @version    : 1.0
 */
#ifndef EXTRACT_SURFACE_JJ_H
#define EXTRACT_SURFACE_JJ_H

#include <Eigen/Core>
#include <string>

template <typename T>
int extract_surface(const Eigen::Matrix<T, -1, -1>& nods, const Eigen::MatrixXi& cells, Eigen::MatrixXi& surface, const std::string& type = "tet");

#endif  // EXTRACT_SURFACE_JJ_H
