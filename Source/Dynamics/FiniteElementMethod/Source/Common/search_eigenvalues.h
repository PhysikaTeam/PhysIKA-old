/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: fast searching some partial eigenvalues
 * @version    : 1.0
 */
#ifndef PhysIKA_SEARCH_EIGENVALUES
#define PhysIKA_SEARCH_EIGENVALUES
// #define EIGEN_USE_LAPACK
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include "Common/error.h"
#include "Common/DEFINE_TYPE.h"

namespace PhysIKA {
using integer = int;

template <typename T>
using mat3 = Eigen::Matrix<T, 3, 3>;

template <typename T>
int eig(const SPM_R<T>& A, MAT<T>& eig_vec, VEC<T>& eig_val);

template <typename T>
int eig_jac(const mat3<T>& mat_A, mat3<T>& eig_vec, Eigen::Matrix<T, 3, 1>& eig_val);

//A is assumed to be symmetric and positive definite
template <typename T>
int get_spectrum(const SPM_R<T>& A, const size_t band, const size_t num_band, MAT<T>& eig_vec, VEC<T>& eig_val);

// template<typename T>
// int eig_jac(const SPM_R<T>& mat_A, MAT<T>&eig_vec, VEC<T>& eig_val);

template <typename MAT_TYPE, typename VEC_TYPE>
int get_dominant_eigenvec(const MAT_TYPE& A, VEC_TYPE& dom_eig_vec, const size_t max_itrs = 1000)
{
    VEC_TYPE v = VEC_TYPE::Random(A.cols());

    double   last_lambda = 1e40, new_lambda = 0;
    VEC_TYPE w(v.size());
    for (size_t i = 0; i < max_itrs; ++i)
    {
        w          = A * v;
        new_lambda = v.dot(w);
        if (fabs(new_lambda - last_lambda) > 1e-10)
            last_lambda = new_lambda;
        else
            break;
        v = w / w.norm();
    }
    dom_eig_vec = v;
    return 0;
}

int find_max_min_eigenvalues(const Eigen::SparseMatrix<double>& A, double& max_eigvalue, double& min_eigvalue);

double find_max_eigenvalue(const Eigen::SparseMatrix<double>& A, const size_t max_itrs = 1000);

double find_min_eigenvalue(const Eigen::SparseMatrix<double>& A, const size_t max_itrs = 1000);

double find_min_eigenvalue(const Eigen::SparseMatrix<double>& A, const double& max_eig, const size_t max_iters);
double find_condition_number(const Eigen::SparseMatrix<double>& A, const size_t max_itrs = 1000);
}  // namespace PhysIKA
#endif
