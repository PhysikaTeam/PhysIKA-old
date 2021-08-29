/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: customized precondition conjugate gradient method
 * @version    : 1.0
 */
#include <iostream>
#include <fstream>

#include "Common/BCSR.h"
#include "customized_pcg.h"
namespace PhysIKA {
using namespace std;
using namespace Eigen;
//TODO: optimize A * b for symmetric case
template <typename T, typename SPM_TYPE>
PCG<T, SPM_TYPE>::PCG(const precond_type<T>& M, const T tol, const bool sym)
    : M_(M), tol_(tol), sym_(sym)
{
}

template <typename T, typename SPM_TYPE>
int PCG<T, SPM_TYPE>::solve(const SPM_TYPE& A, const T* b, T* solution)
{
    const size_t      dim = A.cols();
    Map<const VEC<T>> rhs(b, dim);
    max_itrs_ = dim * 2;
    T tol     = rhs.norm() * tol_;

    VEC<T> xk = VEC<T>::Zero(dim);
    VEC<T> rk = rhs - A * xk;
    VEC<T> dk = M_ == nullptr ? rk : (*M_)(rk);
    VEC<T> rk_p(dim);
    VEC<T> Adk(dim);

    T dkAdk, alpha, beta, residual;

    bool   convergence = false;
    size_t itrs        = 0;
    T      time_whole = 0.0, time_pre = 0.0;

    if (M_ != nullptr)
    {
        __TIME_BEGIN__;
        for (size_t i = 0; i < max_itrs_; ++i)
        {
            __TIME_BEGIN__;
            Adk.noalias()   = A * dk;
            const T time_Ax = __TIME_END__("Ax", false);
            dkAdk           = dk.dot(Adk);
            alpha           = dk.dot(rk) / dkAdk;
            xk.noalias() += alpha * dk;
            rk.noalias() -= alpha * Adk;
            residual = rk.norm();
            if (residual < tol)
            {
                convergence = true;
                itrs        = i;
                break;
            }
            __TIME_BEGIN__;
            rk_p.noalias() = (*M_)(rk);
            time_pre += __TIME_END__("pre", false);
            beta = rk_p.dot(Adk) / dkAdk;
            dk *= -beta;
            dk.noalias() += rk_p;
        }
        time_whole = __TIME_END__("whole", false);
    }
    else
    {
        T rkrk = rk.dot(rk), rkrk_new;
        for (size_t i = 0; i < max_itrs_; ++i)
        {
            Adk.noalias() = A * dk;
            alpha         = rkrk / dk.dot(Adk);
            xk.noalias() += alpha * dk;
            rk.noalias() -= alpha * Adk;
            residual = rk.norm();
            if (residual < tol)
            {
                convergence = true;
                itrs        = i;
                break;
            }
            rkrk_new = rk.dot(rk);
            beta     = rkrk_new / rkrk;

            dk *= beta;
            dk.noalias() += rk;
            rkrk = rkrk_new;
        }
    }
    cout << "pre / whole " << time_pre / time_whole << endl;
    Map<VEC<T>> sol(solution, dim);
    sol = xk;

    if (convergence)
    {
        cout << "PCG converge with " << itrs
             << " steps and the residual is " << residual << endl;
        return 0;
    }
    else
    {
        cout << "PCG exceed max iterations and the rediual now is " << residual << endl;
        return __LINE__;
    }
}
template class PCG<double, Eigen::SparseMatrix<double, Eigen::RowMajor>>;
template class PCG<float, Eigen::SparseMatrix<float, Eigen::RowMajor>>;
template class PCG<double, BCSR<double, 3>>;
template class PCG<float, BCSR<float, 3>>;

}  // namespace PhysIKA
