#ifndef PhysIKA_NEWTON_METHOD
#define PhysIKA_NEWTON_METHOD
#include <memory>
#include<Eigen/SparseCholesky>	
#include <Eigen/SparseLU>
#include "Geometry/embedded_interpolate.h"
#include "Common/framework.h"
#include "Common/DEFINE_TYPE.h"
namespace PhysIKA{

template<typename T>
using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;


template<typename T>
using linear_solver_type = std::function<int(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A,
                                             const T* b,
                                             const Eigen::SparseMatrix<T, Eigen::RowMajor>& J,
                                             const T* c,
                                             T* solution)>;

template<typename T, size_t dim_>
class newton_base: public solver<T, dim_>{
 public:
  newton_base(const std::shared_ptr<Problem<T, dim_>>& pb,
              const size_t max_iter, const T tol,
              const bool line_search,const bool hes_is_constant,
              linear_solver_type<T> linear_solver,
              std::shared_ptr<dat_str_core<T, dim_>> dat_str = nullptr);

  virtual int solve(T* x_star) const;
 protected:
  virtual int solve_linear_eq(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A, const T* b, const Eigen::SparseMatrix<T, Eigen::RowMajor>& J, const T* c, const T* x0, T* solution)const;
  mutable linear_solver_type<T> linear_solver_;

  const bool line_search_;
  const bool hes_is_constant_;


  const T tol_;
  const size_t total_dim_;
  const size_t max_iter_;
  
 protected:
  decltype(solver<T,dim_>::dat_str_)& dat_str_ = solver<T,dim_>::dat_str_;
  decltype(solver<T,dim_>::pb_)& pb_ = solver<T,dim_>::pb_;

  void get_J_C(const T* x, Eigen::SparseMatrix<T, Eigen::RowMajor>& J, VEC<T>& C) const;
      
};


template<typename T, size_t dim_>
class newton_base_with_embedded: public newton_base<T, dim_>{
 public:
  newton_base_with_embedded(
    const std::shared_ptr<Problem<T, dim_>>& pb,
    const size_t max_iter, const T tol,
    const bool line_search,const bool hes_is_constant,
    linear_solver_type<T> linear_solver,
    std::shared_ptr<dat_str_core<T, dim_>> dat_str,
    size_t dof_of_nods,
    std::shared_ptr<embedded_interpolate<T>> embedded_interp)
    : newton_base<T, dim_>(pb, max_iter, tol, line_search, hes_is_constant, linear_solver, dat_str), dof_of_nods_(dof_of_nods), embedded_interp_(embedded_interp) {}
  virtual int solve(T* x_star) const;


  using newton_base<T, dim_>::solve_linear_eq;
  using newton_base<T, dim_>::linear_solver_;
  using newton_base<T, dim_>::line_search_;
  using newton_base<T, dim_>::hes_is_constant_;
  using newton_base<T, dim_>::tol_;
  using newton_base<T, dim_>::total_dim_;
  using newton_base<T, dim_>::max_iter_;
  using newton_base<T, dim_>::dat_str_;
  using newton_base<T, dim_>::pb_;
  using newton_base<T, dim_>::get_J_C;

protected:
  size_t dof_of_nods_;
  std::shared_ptr<embedded_interpolate<T>> embedded_interp_;
};
}
#endif
