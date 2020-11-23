#include <iomanip>
#include "newton_method.h"
#include "Common/config.h"
#include "line_search.h"
#include "Io/io.h"
#include "fast_ms_solver.h"


namespace PhysIKA{
using namespace std;
using namespace Eigen;


  template<typename T>
  Matrix<T, -1, 1> optimize_d(const Matrix<T, 1, -1> &H, const Matrix<T, -1, 1> &length)
  {
    const int edge_num = length.size();
    Matrix<T, -1, 1> d(3*edge_num);
    for (int i = 0; i < edge_num; ++i)
    {
      T len = H.block(0, 3*i, 1, 3).norm();
      for (int j = 0; j < 3; ++j)
        d(3*i + j) = length(3*i) * H(3*i + j) / len;
    }

    return d;
  }

  template<typename T, size_t dim_>
  int fast_ms_solver<T, dim_>::solve(T* x_star) const
  {
    Map<Matrix<T, -1, -1>> X_star(x_star, 3, dof_of_nods_ / 3);
    Matrix<T, -1, -1> X_coarse = embedded_interp_->get_verts();
    Map<VEC<T>> x_coarse(X_coarse.data(), total_dim_);

    if (q_n1_.size() ==  0)
    {
      q_n1_ = x_coarse;
      newton_base_with_embedded<T, dim_>::solve(x_star);

      X_coarse = embedded_interp_->get_verts();
      q_n_ = x_coarse;
    }
    else
    {
      q_n1_ = x_coarse;
      
      Matrix<T, -1, 1> y = 2 * q_n_ - q_n1_;
      T h2 = solver_info_->h_ * solver_info_->h_;
      Matrix<T, -1, 1> b = solver_info_->f_ext_ - h2 * solver_info_->optM_ * y;
      for (int i = 0; i < 10; ++i)
      {
        // optimize d
        Matrix<T, 1, -1> coeff_d = -h2 * x_coarse.transpose() * solver_info_->optJ_;
        Matrix<T, 1, -1> d = optimize_d(coeff_d, solver_info_->origin_length_);
        // optimize x
        Matrix<T, -1, 1> r = h2 * solver_info_->optJ_ * d - b;
        Eigen::SparseMatrix<T, Eigen::RowMajor> J;
        Matrix<T, -1, 1> C;
        __TIME_BEGIN__;
        IF_ERR(return, solve_linear_eq(solver_info_->optX_, r.data(), J, C.data(), x_star, x_coarse.data()));
        __TIME_END__("fast ms solve equation", true);
      }

      q_n_ = x_coarse;
      const SparseMatrix<T> &c2f_coeff = embedded_interp_->get_coarse_to_fine_coeff();
      X_star = X_coarse * c2f_coeff;
      embedded_interp_->set_verts(X_coarse);
    }

    return 0;
  }


  
template class fast_ms_solver<double, 3>;
template class fast_ms_solver<float, 3>;
}
