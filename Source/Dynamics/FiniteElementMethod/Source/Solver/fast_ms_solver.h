/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: fast mass spring solver
 * @version    : 1.0
 */
#ifndef PhysIKA_FAST_MS_SOLVER_JJ_H
#define PhysIKA_FAST_MS_SOLVER_JJ_H
#include <memory>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include "Geometry/embedded_interpolate.h"
#include "Common/framework.h"
#include "Common/DEFINE_TYPE.h"
#include "semi_implicit_euler.h"
#include "newton_method.h"

namespace PhysIKA {

/**
 * fast mass spring method infos.
 *
 */
template <typename T = float>
class fast_ms_info
{
public:
    Eigen::SparseMatrix<T>  optL_;
    Eigen::SparseMatrix<T>  optJ_;
    Eigen::SparseMatrix<T>  optM_;
    Eigen::SparseMatrix<T>  optX_;  // quad coefficient of X
    Eigen::Matrix<T, -1, 1> f_ext_;
    Eigen::Matrix<T, -1, 1> origin_length_;
    T                       h_;
    int                     edge_num_;
};

/**
 * fast mass spring solver.
 *
 */
template <typename T, size_t dim_>
class fast_ms_solver : public newton_base_with_embedded<T, dim_>
{
public:
    fast_ms_solver(
        const std::shared_ptr<Problem<T, dim_>>& pb,
        const size_t                             max_iter,
        const T                                  tol,
        const bool                               line_search,
        const bool                               hes_is_constant,
        linear_solver_type<T>                    linear_solver,
        std::shared_ptr<dat_str_core<T, dim_>>   dat_str,
        size_t                                   dof_of_nods,
        std::shared_ptr<embedded_interpolate<T>> embedded_interp,
        std::shared_ptr<semi_implicit<T>>        semi,
        std::shared_ptr<fast_ms_info<T>>         solver_info)
        : newton_base_with_embedded<T, dim_>(pb, max_iter, tol, line_search, hes_is_constant, linear_solver, dat_str, dof_of_nods, embedded_interp, semi), solver_info_(solver_info) {}

    virtual int solve(T* x_star) const;

    using newton_base_with_embedded<T, dim_>::solve_linear_eq;
    using newton_base_with_embedded<T, dim_>::linear_solver_;
    using newton_base_with_embedded<T, dim_>::line_search_;
    using newton_base_with_embedded<T, dim_>::hes_is_constant_;
    using newton_base_with_embedded<T, dim_>::tol_;
    using newton_base_with_embedded<T, dim_>::total_dim_;
    using newton_base_with_embedded<T, dim_>::max_iter_;
    using newton_base_with_embedded<T, dim_>::dat_str_;
    using newton_base_with_embedded<T, dim_>::pb_;
    using newton_base_with_embedded<T, dim_>::get_J_C;
    using newton_base_with_embedded<T, dim_>::dof_of_nods_;
    using newton_base_with_embedded<T, dim_>::embedded_interp_;

protected:
    std::shared_ptr<fast_ms_info<T>> solver_info_;
    static Eigen::Matrix<T, -1, 1>   q_n1_;
    static Eigen::Matrix<T, -1, 1>   q_n_;
    mutable Eigen::Matrix<T, -1, 1>  y_;
};
}  // namespace PhysIKA
#endif
