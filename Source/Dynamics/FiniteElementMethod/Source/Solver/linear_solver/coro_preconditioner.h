/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: precoditioner for co-rotational solver
 * @version    : 1.0
 */
#ifndef CORO_PRECONDITIONER
#define CORO_PRECONDITIONER

#include <Eigen/SparseCholesky>

#include "Common/search_eigenvalues.h"
#include "Common/diag_BCSR.h"
#include "Model/fem/elas_energy.h"

#include "preconditioner.h"

template <typename T>
using ldlt_type = Eigen::SimplicialLDLT<SPM_C<T>>;

template <typename T>
using ldlt_ptr = std::shared_ptr<ldlt_type<T>>;

template <typename T>
using elas_intf_ptr = std::shared_ptr<PhysIKA::elas_intf<T, 3>>;

namespace PhysIKA {
/**
 * corotated preconditioner class
 *
 */
template <typename T>
class coro_preconditioner : public preconditioner<T>
{
public:
    coro_preconditioner(const ldlt_ptr<T>& fac_hes_rest, const std::shared_ptr<diag_BCSR<T, 3>>& R)
        : fac_hes_rest_(fac_hes_rest), R_(R), RT_(R_->transpose()) {}

    /*use R(x) A_bar R(x)T to approximate A(x), where A_bar is the hessian at the rest shape,
   R(x) is the assembled rotation matrix for each vertex.*/
    VEC<T> operator()(const VEC<T>& r) const
    {
        VEC<T> RT_r   = RT_ * r;
        VEC<T> A_RT_r = fac_hes_rest_->solve(RT_r);
        return (*R_) * A_RT_r;
    };

protected:
    const ldlt_ptr<T>                      fac_hes_rest_;
    const std::shared_ptr<diag_BCSR<T, 3>> R_;
    const diag_BCSR<T, 3>                  RT_;
};

/**
 * factory for generating coro preconditioner
 *
 */
template <typename T>
class coro_precond_fac
{
public:
    coro_precond_fac(const ldlt_ptr<T>& fac_hes_rest)
        : fac_hes_rest_(fac_hes_rest), R_(std::make_shared<diag_BCSR<T, 3>>())
    {
        exit_if(fac_hes_rest_->info() != Eigen::Success, "Rest hessian can not perform LDLT.");
    }

    std::shared_ptr<coro_preconditioner<T>> build_preconditioner() const
    {
        exit_if(R_->size() == 0 || fac_hes_rest_ == nullptr);
        return std::make_shared<coro_preconditioner<T>>(fac_hes_rest_, R_);
    }
    std::shared_ptr<diag_BCSR<T, 3>> get_R() const
    {
        return R_;
    }

private:
    const ldlt_ptr<T> fac_hes_rest_;

protected:
    std::shared_ptr<diag_BCSR<T, 3>> R_;
    VEC_MAT<MAT3<T>>                 R_per_nod_;
};

/**
 * geometry multigrid based coro preconditioner factory.
 *
 */
template <typename T>
class gmg_coro_precond_fac : public coro_precond_fac<T>
{
public:
    gmg_coro_precond_fac(const ldlt_ptr<T>& fac_hes_rest, const elas_intf_ptr<T>& elas)
        : coro_precond_fac<T>(fac_hes_rest), elas_(elas) {}
    int opt_R(const T* x)
    {
        IF_ERR(return, elas_->aver_ele_R(x, this->R_per_nod_));
        this->R_->setFromDiagMat(this->R_per_nod_);
        return 0;
    }

private:
    const elas_intf_ptr<T> elas_;
};

/**
 * algebraic multigrid based coro preconditioner factory.
 *
 */
template <typename T>
class amg_coro_precond_fac : public coro_precond_fac<T>
{
public:
    //TODO: maybe setting A_bar_diag as argument  is better than get_block_diagonal each time
    amg_coro_precond_fac(const ldlt_ptr<T>& fac_hes_rest, const SPM_R<T>& hes_rest)
        : coro_precond_fac<T>(fac_hes_rest), A_bar_diag_(get_block_diagonal(hes_rest)) {}
    int opt_R(const SPM_R<T>& A)
    {
        IF_ERR(return, minimize_norm_of_diff(A));
        this->R_->setFromDiagMat(this->R_per_nod_);
        return 0;
    }

private:
    VEC_MAT<MAT3<T>>       A_diag_;
    const VEC_MAT<MAT3<T>> A_bar_diag_;

    int minimize_norm_of_diff(const SPM_R<T>& A)
    {
        A_diag_ = get_block_diagonal(A);
        this->R_per_nod_.resize(A_diag_.size());
        const size_t size = A_diag_.size();
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            MAT3<T> A_i     = A_diag_[i];
            MAT3<T> A_bar_i = A_bar_diag_[i];

            MAT3<T> U_i = A_i;
            VEC3<T> E_i = VEC3<T>::Zero();
            eig_jac(A_bar_i, U_i, E_i);

            MAT3<T> U_bar_i;
            VEC3<T> E_bar_i;
            eig_jac(A_bar_i, U_bar_i, E_bar_i);
            this->R_per_nod_[i] = U_i * U_bar_i.transpose();
        }
        return 0;
    }
};

}  // namespace PhysIKA
#endif
