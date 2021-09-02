/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: edge energy's gradient and hessian for mass spring method.
 * @version    : 1.0
 */
#ifndef EDGE_GRADIENT_AND_HESSIAN_JJ_H
#define EDGE_GRADIENT_AND_HESSIAN_JJ_H

#include <Eigen/Core>

#include "edge_gradient.h"
#include "edge_energy.h"
#include "edge_hessian.h"

template <typename T>
T GetEdgeEnergy(T* __restrict x, T K, T l0)
{
    T E;
    EdgeEnergy(x, &K, &l0, &E);
    return E;
}

template <typename T>
Eigen::Matrix<T, 6, 1> GetEdgeGradient(T* __restrict x, T K, T l0)
{
    Eigen::Matrix<T, 6, 1> g = Eigen::Matrix<T, 6, 1>::Zero();
    EdgeGradient(x, &K, &l0, &g[0]);
    return g;
}

template <typename T>
Eigen::Matrix<T, 6, 6> GetEdgeHessian(T* __restrict x, T K, T l0)
{
    Eigen::Matrix<T, 6, 6> H = Eigen::Matrix<T, 6, 6>::Zero();
    EdgeHessian(x, &K, &l0, &H(0, 0));
    return H;
}

template float  GetEdgeEnergy<float>(float* __restrict x, float K, float l0);
template double GetEdgeEnergy<double>(double* __restrict x, double K, double l0);

template Eigen::Matrix<float, 6, 1>  GetEdgeGradient<float>(float* __restrict x, float K, float l0);
template Eigen::Matrix<double, 6, 1> GetEdgeGradient<double>(double* __restrict x, double K, double l0);

template Eigen::Matrix<float, 6, 6>  GetEdgeHessian<float>(float* __restrict x, float K, float l0);
template Eigen::Matrix<double, 6, 6> GetEdgeHessian<double>(double* __restrict x, double K, double l0);

#endif  // EDGE_GRADIENT_AND_HESSIAN_JJ_H
