/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: co-rotation solver.
 * @version    : 1.0
 */
#ifndef PhysIKA_CORO_SOLVER
#define PhysIKA_CORO_SOLVER

#include <Eigen/Eigenvalues>

#include "Solver/linear_solver/customized_pcg.h"
#include "Solver/linear_solver/coro_preconditioner.h"
#include "newton_method.h"

namespace PhysIKA {
/**
 * geometry multigrid based coro solver.
 *
 */
template <typename T>
class gmg_coro_solver : public newton_base<T, 3>
{
public:
    gmg_coro_solver(const std::shared_ptr<Problem<T, 3>>& pb,
                    const size_t                          max_iter,
                    const T                               tol,
                    const bool                            line_search,
                    const bool                            hes_is_constant,
                    const std::shared_ptr<ldlt_type<T>>&  fac_hes_rest,
                    std::shared_ptr<elas_intf<T, 3>>&     elas_intf_ptr,
                    const SPM_R<T>                        hes_rest,
                    const T                               cg_tol  = 1e-3,
                    std::shared_ptr<dat_str_core<T, 3>>   dat_str = nullptr)
        : newton_base<T, 3>(pb, max_iter, tol, line_search, hes_is_constant, nullptr, dat_str), pd_fac_(make_shared<gmg_coro_precond_fac<T>>(fac_hes_rest, elas_intf_ptr)), cg_tol_(cg_tol), hes_rest_(hes_rest) {}

public:
    std::shared_ptr<gmg_coro_precond_fac<T>> pd_fac_;
    const T                                  cg_tol_;

    //add this member for testing spectrum, remove this later
    const SPM_R<T> hes_rest_;

    /*
   When  kinetic_ = nullptr and w_pos = 0, A = K.
   We would like to check
   || U^TK(x)U - \Lambda ||
   where U^T K_tild
e U = \Lambda_tilde and \Lambda is the eigenavlues of K(x).

   U is caculated by
   U = R U_bar^T where U_bar^T is the transpose of the eigenvector of K_bar (rest shape).
  */
    int test_spectrum(const SPM_R<T>& A, const size_t band, const size_t num_band, MAT<T>& eig_vec) const
    {
        printf("test spectrum.\n");
        const size_t num_eig = band * num_band;
        const size_t removed = band;

        MAT<T> eig_vec_rest;
        VEC<T> eig_val_rest;

        IF_ERR(return, get_spectrum(hes_rest_, band, num_band, eig_vec_rest, eig_val_rest));
        eig_vec_rest = eig_vec_rest.rightCols(num_eig - removed).eval();
        eig_val_rest = eig_val_rest.bottomRows(num_eig - removed).eval();

        const auto R_ = pd_fac_->get_R();

#if 0
    SPM_R<T> A_tilde  = R_eigen * hes_rest_ * R_eigen.transpose();{
      SPM_C<T> R_eigen(R_->rows(), R_->cols());{
        vector<Triplet<T>> trips;
        for(size_t i = 0; i < R_->rowsb(); ++i){
          const auto R_i = (*R_)(i);
          for(size_t m = 0; m < 3; ++m)
            for(size_t n = 0; n < 3; ++n){
              trips.push_back(Triplet<T>(3 * i + m, 3 * i +  n, R_i(m, n)));
            }
        }
        R_eigen.reserve(trips.size());
        R_eigen.setFromTriplets(trips.begin(), trips.end());
      }
    }
    printf("|| A - A_tilde || = %f \n", (A - A_tilde).norm());
#endif

        MAT<T> eig_vec_tilde(eig_vec_rest.rows(), eig_vec_rest.cols());
#pragma omp parallel for
        for (size_t i = 0; i < eig_vec_rest.cols(); ++i)
        {
            eig_vec_tilde.col(i) = (*R_) * eig_vec_rest.col(i);
        }

        eig_vec;
        VEC<T> eig_val;
        // eig(A, eig_vec, eig_val);
        IF_ERR(return, get_spectrum(A, band, num_band, eig_vec, eig_val));
        eig_val = eig_val.bottomRows(num_eig - removed).eval();
        cout << "eig val of A \n"
             << eig_val << endl;

        const MAT<T> eig_val_diag = eig_val.asDiagonal();

        const MAT<T> diff = eig_vec_tilde.transpose() * A * eig_vec_tilde - eig_val_diag;
        cout << "diff \n"
             << diff.diagonal() << endl;
        T norm = diff.norm();
        printf("|| U^TK(x)U - \\Lambda || = %f \n", norm);
        return 0;
    }

    int solve_linear_eq(const SPM_R<T>& A, const T* b, const SPM_R<T>& J, const T* c, const T* x0, T* solution) const override
    {

        pd_fac_->opt_R(x0);
        auto coro_pcg_solver = make_shared<PCG<T, SPM_R<T>>>(pd_fac_->build_preconditioner(), cg_tol_);
        return coro_pcg_solver->solve(A, b, solution);
    }
};

/**
 * algebraic multigrid based coro solver.
 *
 */
template <typename T>
class amg_coro_solver : public newton_base<T, 3>
{
public:
    amg_coro_solver(const std::shared_ptr<Problem<T, 3>>& pb,
                    const size_t                          max_iter,
                    const T                               tol,
                    const bool                            line_search,
                    const bool                            hes_is_constant,
                    const SPM<T>&                         hes_rest,
                    const std::shared_ptr<ldlt_type<T>>&  fac_hes_rest,
                    const T                               cg_tol  = 1e-3,
                    std::shared_ptr<dat_str_core<T, 3>>   dat_str = nullptr)
        : newton_base<T, 3>(pb, max_iter, tol, line_search, hes_is_constant, nullptr, dat_str), pd_fac_(make_shared<amg_coro_precond_fac<T>>(fac_hes_rest, hes_rest)), cg_tol_(cg_tol) {}

private:
    std::shared_ptr<amg_coro_precond_fac<T>> pd_fac_;
    const T                                  cg_tol_;
    int                                      solve_linear_eq(const SPM_R<T>& A, const T* b, const SPM_R<T>& J, const T* c, const T* x0, T* solution) const override
    {
        pd_fac_->opt_R(A);
        auto coro_pcg_solver = make_shared<PCG<T, SPM_R<T>>>(pd_fac_->build_preconditioner(), cg_tol_);
        return coro_pcg_solver->solve(A, b, solution);
    }
};

}  // namespace PhysIKA
#endif
