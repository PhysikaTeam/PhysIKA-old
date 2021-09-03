#pragma once

#include <Eigen/Core>
#include <vector>

template <typename T>
void HexDiv1To8(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 27>& NewP, Eigen::Matrix<int, 8, 8>& C);

template <typename T>
inline int getPIdx(Eigen::Matrix<T, 3, 1>& p, std::vector<Eigen::Matrix<T, 3, 1>>& P);

template <typename T>
void HexDiv1To64(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 125>& NewP, Eigen::Matrix<int, 8, 64>& C);
