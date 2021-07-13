/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: newton method for optimization problem.
 * @version    : 1.0
 */
#include <iomanip>
#include "newton_method.h"
#include "Common/config.h"
#include "line_search.h"
#include "Io/io.h"

namespace PhysIKA {
using namespace std;
using namespace Eigen;

template <typename T, size_t dim_>
newton_base<T, dim_>::newton_base(
    const std::shared_ptr<Problem<T, dim_>>& pb,
    const size_t                             max_iter,
    const T                                  tol,
    const bool                               line_search,
    const bool                               hes_is_constant,
    linear_solver_type<T>                    linear_solver,
    std::shared_ptr<dat_str_core<T, dim_>>   dat_str,
    std::shared_ptr<semi_implicit<T>>        semi)
    : solver<T, dim_>(pb, dat_str), max_iter_(max_iter), tol_(tol), line_search_(line_search), hes_is_constant_(hes_is_constant), linear_solver_(linear_solver), total_dim_(pb->Nx()), semi_implicit_(semi)
{
}

template <typename T, size_t dim_>
int newton_base<T, dim_>::solve_linear_eq(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A, const T* b, const Eigen::SparseMatrix<T, Eigen::RowMajor>& J, const T* c, const T* x0, T* solution) const
{
    return linear_solver_(A, b, J, c, solution);
}

template <typename T, size_t dim_>
void newton_base<T, dim_>::get_J_C(const T* x, Eigen::SparseMatrix<T, Eigen::RowMajor>& J, VEC<T>& C) const
{
    if (pb_->constraint_ == nullptr || pb_->constraint_->Nf() == 0)
    {
        J.resize(0, 0);
        return;
    }

    C             = VEC<T>::Zero(total_dim_);
    VEC<T> zero_x = VEC<T>::Zero(total_dim_);
    pb_->constraint_->Val(x, C.data());
    C *= -1;

    vector<Triplet<T>> trips;
    auto               cons = pb_->constraint_;
    cons->Jac(x, 0, &trips);
    J.resize(cons->Nf(), cons->Nx());
    J.reserve(trips.size());
    J.setFromTriplets(trips.begin(), trips.end());
    return;
}

template <typename T, size_t dim_>
int newton_base<T, dim_>::solve(T* x_star) const
{
    Map<VEC<T>> X_star(x_star, total_dim_);

    VEC<T> res           = VEC<T>::Zero(total_dim_);
    VEC<T> solution      = VEC<T>::Zero(total_dim_);
    T      res_last_iter = 9999999;

    for (size_t newton_i = 0; newton_i < max_iter_; ++newton_i)
    {
        cout << "[INFO]>>>>>>>>>>>>>>>newton iter is " << newton_i << endl;
        dat_str_->set_zero();
        IF_ERR(return, pb_->energy_->Val(x_star, dat_str_));
        IF_ERR(return, pb_->energy_->Gra(x_star, dat_str_));
        res = -dat_str_->get_gra();
        if (!hes_is_constant_)
        {
            IF_ERR(return, pb_->energy_->Hes(x_star, dat_str_));
            dat_str_->hes_compress();
        }

        const T res_value = res.array().abs().sum();
        cout << "[INFO]: energy gradient " << res_value << endl;
        cout << "[INFO]: energy value: " << dat_str_->get_val() << endl;

        if (res_value < tol_)
        {
            cout << endl;
            break;
        }

        // if(newton_i > 1 && fabs(res_value / res_last_iter -1) < 1e-2 )
        //   break;

        res_last_iter = res_value;

        const auto                              A = dat_str_->get_hes();
        Eigen::SparseMatrix<T, Eigen::RowMajor> J;
        Matrix<T, -1, 1>                        C;
        get_J_C(x_star, J, C);
        __TIME_BEGIN__;
        IF_ERR(return, solve_linear_eq(A, res.data(), J, C.data(), x_star, solution.data()));
        __TIME_END__("solve eq", true);
        if (line_search_)
            X_star += line_search<T, dim_>(dat_str_->get_val(), static_cast<T>(res.dot(solution)), pb_->energy_, dat_str_, x_star, solution.data()) * solution;
        else
            X_star += solution;
    }
    return 0;
}

template <typename T, size_t dim_>
int newton_base_with_embedded<T, dim_>::solve(T* x_star) const
{
    Map<Matrix<T, -1, -1>> X_star(x_star, 3, dof_of_nods_ / 3);

    Matrix<T, -1, -1> X_coarse = embedded_interp_->get_verts();
    Map<VEC<T>>       x_coarse(X_coarse.data(), total_dim_);

    if (semi_implicit_ != nullptr)
    {
        dat_str_->set_zero();
        IF_ERR(return, pb_->energy_->Val(x_coarse.data(), dat_str_));
        IF_ERR(return, pb_->energy_->Gra(x_coarse.data(), dat_str_));
        Matrix<T, -1, 1> jaccobi         = -dat_str_->get_gra();
        x_coarse                         = semi_implicit_->solve(jaccobi);
        const SparseMatrix<T>& c2f_coeff = embedded_interp_->get_coarse_to_fine_coeff();
        X_star                           = X_coarse * c2f_coeff;
        embedded_interp_->set_verts(X_coarse);
        cerr << "[  \033[1;31merror\033[0m  ] "
             << "norm of X_star:" << X_star.norm() << endl;
        return 0;
    }

    VEC<T> res           = VEC<T>::Zero(total_dim_);
    VEC<T> solution      = VEC<T>::Zero(total_dim_);
    T      res_last_iter = 9999999;

    for (size_t newton_i = 0; newton_i < max_iter_; ++newton_i)
    {
        cout << "[INFO]>>>>>>>>>>>>>>>newton iter is " << newton_i << endl;
        dat_str_->set_zero();
        IF_ERR(return, pb_->energy_->Val(x_coarse.data(), dat_str_));
        IF_ERR(return, pb_->energy_->Gra(x_coarse.data(), dat_str_));
        res = -dat_str_->get_gra();
        if (!hes_is_constant_)
        {
            IF_ERR(return, pb_->energy_->Hes(x_coarse.data(), dat_str_));
            dat_str_->hes_compress();
        }

        const T res_value = res.array().abs().sum();
        cout << "[INFO]: energy gradient " << res_value << endl;
        cout << "[INFO]: energy value: " << dat_str_->get_val() << endl;

        if (res_value < tol_)
        {
            cout << endl;
            break;
        }

        // if(newton_i > 1 && fabs(res_value / res_last_iter -1) < 1e-2 )
        //   break;

        res_last_iter = res_value;

        const auto A = dat_str_->get_hes();
        cout << "norm of A is " << std::setprecision(20) << A.norm() << endl;
        Eigen::SparseMatrix<T, Eigen::RowMajor> J;
        Matrix<T, -1, 1>                        C;
        get_J_C(x_coarse.data(), J, C);
        __TIME_BEGIN__;
        IF_ERR(return, solve_linear_eq(A, res.data(), J, C.data(), x_coarse.data(), solution.data()));
        __TIME_END__("solve eq", true);
        if (line_search_)
            x_coarse += line_search<T, dim_>(dat_str_->get_val(), static_cast<T>(res.dot(solution)), pb_->energy_, dat_str_, x_coarse.data(), solution.data()) * solution;
        else
            x_coarse += solution;
    }

    cerr << "[   \033[1;37mlog\033[0m   ] "
         << "len of pos:" << solution.norm() << endl;
    const SparseMatrix<T>& c2f_coeff = embedded_interp_->get_coarse_to_fine_coeff();

    X_star = X_coarse * c2f_coeff;
    embedded_interp_->set_verts(X_coarse);

    return 0;
}

template class newton_base<double, 3>;
template class newton_base<float, 3>;
template class newton_base_with_embedded<double, 3>;
template class newton_base_with_embedded<float, 3>;
}  // namespace PhysIKA
