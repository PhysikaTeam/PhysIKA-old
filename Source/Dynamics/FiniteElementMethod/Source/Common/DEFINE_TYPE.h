/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: define some type usage
 * @version    : 1.0
 */
#ifndef PhysIKA_DEFINE_TYPE
#define PhysIKA_DEFINE_TYPE
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

// using REAL = float;

// #define
template <typename T>
inline constexpr T scalar_max()
{
    return std::numeric_limits<T>::max() / 10;
}
template <typename T>
inline constexpr T scalar_eps()
{
    return 1;
}

template <>
inline constexpr double scalar_eps()
{
    return 1e-9f;
}
template <>
inline constexpr float scalar_eps()
{
    return 1e-7f;
}

//type for matrix and vector
template <typename T>
using VEC = Eigen::Matrix<T, -1, 1>;

template <typename T>
using VEC3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using MAT = Eigen::Matrix<T, -1, -1>;

template <typename T>
using MAT3 = Eigen::Matrix<T, 3, 3>;

template <typename T>
using SPM_R = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <typename T>
using SPM_C = Eigen::SparseMatrix<T, Eigen::ColMajor>;

#endif
