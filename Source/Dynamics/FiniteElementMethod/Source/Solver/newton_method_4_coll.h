#ifndef PhysIKA_NEWTON_METHOD_FOR_COLLISION
#define PhysIKA_NEWTON_METHOD_FOR_COLLISION

#define USE_STOPWATCH_HELPER_FLAG
#define INFO_PRINT_FLAG
#define ERROR_PRINT_FLAG
#define WARN_PRINT_FLAG
#define DEBUG_PRINT_FLAG

#include "Common/def.h"
#include "newton_method.h"
#include <iostream>
#include "Problem/constraint/constraints.h"
#include "Common/timer/timer_utils.h"
namespace PhysIKA{
template<typename T, size_t dim_>
class newton_4_coll : public newton_base<T, dim_>{
 public:
  newton_4_coll(const std::shared_ptr<Problem<T, dim_>>& pb,
                const size_t max_iter, const T tol,
                const bool line_search,const bool constant_hes,
                linear_solver_type<T>& linear_solver,
                std::shared_ptr<constraint_4_coll<T>>& coll,
                linear_solver_type<T>& kkt,
                std::shared_ptr<dat_str_core<T, dim_>> dat_str = nullptr):
      newton_base<T,dim_>(pb, max_iter, tol, line_search,
                          constant_hes, linear_solver, dat_str),coll_(coll), kkt_(kkt), ls_(linear_solver){
  }


  int solve(T* x_star) const{
    chaos::utils::STW_START("newton_solve");
    Eigen::Map<Eigen::Matrix<T, -1, 1>> X_new(x_star, newton_base<T, dim_>::pb_->Nx());
    const Eigen::Matrix<T, -1 ,1> X_old = X_new;
    newton_base<T, dim_>::linear_solver_ = this->ls_;
    newton_base<T, dim_>::solve(x_star);
    bool res = coll_->verify_no_collision(x_star);
    if (!res) {
      newton_base<T, dim_>::linear_solver_ = this->kkt_;
      X_new = X_old;
      newton_base<T, dim_>::solve(x_star);
    }
    chaos::utils::STW_END("newton_solve", "newton_solve");
    return 0;
  }

  int solve_linear_eq(const SPM<T>& A, const T* b, const SPM<T>& J, const T* c, const T* x0, T* solution) const{
    if(J.rows() != 0)
      return kkt_(A, b, J, c, solution);
    else
      return ls_(A, b, J, c, solution);
      // return newton_base<T, dim_>::linear_solver_(A, b, J, c, solution);
  }
  private:
  linear_solver_type<T> kkt_;
  linear_solver_type<T> ls_;
  std::shared_ptr<constraint_4_coll<T>> coll_;
};
}
#endif
