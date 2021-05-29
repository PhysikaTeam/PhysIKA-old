#ifndef EXTRACT_SURFACE_JJ_H
#define EXTRACT_SURFACE_JJ_H

#include <Eigen/Core>
#include <string>


template<typename T>
int extract_surface(const Eigen::Matrix<T, -1, -1> &nods, const Eigen::MatrixXi &cells, Eigen::MatrixXi &surface, const std::string &type = "tet");


#endif // EXTRACT_SURFACE_JJ_H
