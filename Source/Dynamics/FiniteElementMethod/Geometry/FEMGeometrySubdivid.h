#pragma once

#include <Eigen/Core>
#include <vector>

/**
 * @brief Hexahedron div 1 to 8
 * 
 * @tparam T 
 * @param P 
 * @param NewP 
 * @param C 
 */
template <typename T>
void HexDiv1To8(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 27>& NewP, Eigen::Matrix<int, 8, 8>& C);

/**
 * @brief Get the point index
 * 
 * @tparam T 
 * @param p 
 * @param P 
 * @return int 
 */
template <typename T>
inline int getPIdx(Eigen::Matrix<T, 3, 1>& p, std::vector<Eigen::Matrix<T, 3, 1>>& P);

/**
 * @brief Hexahedron div 1 to 64
 * 
 * @tparam T 
 * @param P 
 * @param NewP 
 * @param C 
 */
template <typename T>
void HexDiv1To64(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 125>& NewP, Eigen::Matrix<int, 8, 64>& C);
