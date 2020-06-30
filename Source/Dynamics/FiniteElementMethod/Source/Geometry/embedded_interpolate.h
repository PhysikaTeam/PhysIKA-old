#ifndef EMBEDDED_INTERPOLATE_JJ_H
#define EMBEDDED_INTERPOLATE_JJ_H

#include <Eigen/SparseCore>
#include <memory>
#include <iostream>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>

template<typename T>
class embedded_interpolate
{
public:
  embedded_interpolate(
    Eigen::Matrix<T, -1, -1> &v,
    Eigen::SparseMatrix<T> &c2f_coeff,
    Eigen::SparseMatrix<T> &f2c_coeff,
    Eigen::SparseMatrix<T> &energy_hessian,
    T alpha = 0.586803)
    : verts_(v), coarse_to_fine_coeff_(c2f_coeff), fine_to_coarse_coeff_(f2c_coeff), energy_hessian_(energy_hessian), alpha_(alpha)
    {
      const Eigen::Matrix<T, 3, 3> Id = Eigen::Matrix<T, 3, 3>::Identity();
      c2f_ = Eigen::kroneckerProduct(coarse_to_fine_coeff_, Id);
      c2f_ = c2f_.transpose().eval();

      Eigen::SparseMatrix<T> cffc = c2f_.transpose() * c2f_;
      Eigen::SparseMatrix<T> A = alpha * cffc + energy_hessian_;
      
      ldlt_solver_.compute(A);
      if (ldlt_solver_.info() != Eigen::Success)
      {
        std::cerr << __LINE__ << "error in ldlt solver" << std::endl;
        exit(1);
      }
    }

  int update_verts(const T *fine, int v_num)
    {
      Eigen::Map<const Eigen::Matrix<T, -1, -1>> fine_verts(fine, 3, v_num);
      Eigen::Matrix<T, -1, -1> X_fine_delta = fine_verts - verts_ * coarse_to_fine_coeff_;
      Eigen::Map<Eigen::Matrix<T, -1, -1>>
        x_fine_delta_vec(X_fine_delta.data(), 3*v_num, 1);
      
      Eigen::Matrix<T, -1, -1> b = alpha_ * c2f_.transpose() * x_fine_delta_vec;
      Eigen::Matrix<T, -1, -1> s = ldlt_solver_.solve(b);
      Eigen::Map<Eigen::Matrix<T, -1, -1>> s_vec(s.data(), 3, s.size()/3);

      verts_ += s_vec;
    }

  const Eigen::SparseMatrix<T>& get_coarse_to_fine_coeff() const {
    return coarse_to_fine_coeff_;}
  const Eigen::Matrix<T, -1, -1>& get_verts() const { return verts_; }
  void set_verts(const Eigen::Matrix<T, -1, -1> &v) { verts_ = v; }
private:
  Eigen::Matrix<T, -1, -1> verts_;
  Eigen::SparseMatrix<T> c2f_;
  Eigen::SparseMatrix<T> coarse_to_fine_coeff_;
  Eigen::SparseMatrix<T> fine_to_coarse_coeff_;
  Eigen::SparseMatrix<T> energy_hessian_;
  Eigen::SparseLU<Eigen::SparseMatrix<T>> ldlt_solver_;
  T alpha_;
};


#endif // EMBEDDED_INTERPOLATE_JJ_H
