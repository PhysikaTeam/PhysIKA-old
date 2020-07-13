#ifndef EDGE_GRADIENT_AND_HESSIAN_JJ_H
#define EDGE_GRADIENT_AND_HESSIAN_JJ_H


#include <Eigen/Core>

extern "C"
{
  void edgeenergyfloat(float *x, float *K, float *l0, float *E);
  void edgeenergydouble(double *x, double *K, double *l0, double *E);

  void edgegradientfloat(float *x, float *K, float *l0, float *g);
  void edgegradientdouble(double *x, double *K, double *l0, double *g);

  void edgehessianfloat(float *x, float *K, float *l0, float *H);
  void edgehessiandouble(double *x, double *K, double *l0, double *H);
}

template<typename T>
T GetEdgeEnergy(T *x, T K, T l0)
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value);
  return 0;
}

template<>
float GetEdgeEnergy<float>(float *x, float K, float l0)
{
  float E = 0;
  edgeenergyfloat(x, &K, &l0, &E);
  return E;
}

template<>
double GetEdgeEnergy<double>(double *x, double K, double l0)
{
  double E = 0;
  edgeenergydouble(x, &K, &l0, &E);
  return E;
}

template<typename T>
Eigen::Matrix<T, 6, 1> GetEdgeGradient(T *x, T K, T l0)
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value);
  return Eigen::Matrix<T, 6, 1>::Zero();
}

template<>
Eigen::Matrix<float, 6, 1> GetEdgeGradient<float>(float *x, float K, float l0)
{
  Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
  edgegradientfloat(x,  &K, &l0, &g[0]);
  return g;
}

template<>
Eigen::Matrix<double, 6, 1> GetEdgeGradient<double>(double *x, double K, double l0)
{
  Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
  edgegradientdouble(x,  &K, &l0, &g[0]);
  return g;
}

template<typename T>
Eigen::Matrix<T, 6, 6> GetEdgeHessian(T *x, T K, T l0)
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value);
  return Eigen::Matrix<T, 6, 6>::Zero();
}

template<>
Eigen::Matrix<float, 6, 6> GetEdgeHessian<float>(float *x, float K, float l0)
{
  Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
  edgehessianfloat(x,  &K, &l0, &H(0, 0));
  return H;
}

template<>
Eigen::Matrix<double, 6, 6> GetEdgeHessian<double>(double *x, double K, double l0)
{
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  edgehessiandouble(x, &K, &l0, &H(0, 0));
  return H;
}

template float GetEdgeEnergy<float>(float *x, float K, float l0);
template double GetEdgeEnergy<double>(double *x, double K, double l0);

template Eigen::Matrix<float, 6, 1> GetEdgeGradient<float>(float *x, float K, float l0);
template Eigen::Matrix<double, 6, 1> GetEdgeGradient<double>(double *x,double K, double l0);

template Eigen::Matrix<float, 6, 6> GetEdgeHessian<float>(float *x, float K, float l0);
template Eigen::Matrix<double, 6, 6> GetEdgeHessian<double>(double *x, double K, double l0);

#endif // EDGE_GRADIENT_AND_HESSIAN_JJ_H
