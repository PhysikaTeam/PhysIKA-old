#pragma once
#include <Eigen/Sparse>
#include <Eigen/src/Core/util/Constants.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsBase.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <iostream>
#include <Eigen/Sparse>

namespace PhysIKA {
/**
   * @brief calculate the material consistent coefficients
   * 
   * Eigen::SparseMatrix<double, Eigen::RowMajor> K =
   *    chaos::poisson2d_on_regular_grid(std::stoi(argv[1]), std::stoi(argv[2]));
   * Eigen::SparseMatrix<double, Eigen::RowMajor> K2 =
   *    chaos::poisson2d_on_regular_grid(std::stoi(argv[1]), std::stoi(argv[2])) *
   *    10;
   * Eigen::Matrix<double, -1, 1> M(K.cols());
   * M.setOnes();
   * cout << k_consistent<double, Eigen::RowMajor>(K, M, K2, M) << endl; 
   * 
   * @tparam Real 
   * @tparam Options 
   * @param Kref the reference object stiffness matrix.
   * @param Mref the reference object mass matrix diagonal.
   * @param Kdest the destination object stiffness matrix.
   * @param Mdest the destination object mass matrix diagonal.
   * @param num the number of low frequence to use for calculation.
   * @return Real the coefficients about two objects.
   */
template <typename Real, int Options = Eigen::RowMajor>
Real k_consistent(const Eigen::SparseMatrix<Real, Options>& Kref,
                  const Eigen::Matrix<Real, -1, 1>&         Mref,
                  const Eigen::SparseMatrix<Real, Options>& Kdest,
                  const Eigen::Matrix<Real, -1, 1>&         Mdest,
                  int                                       start = 6,
                  int                                       num   = 6);
}  // namespace PhysIKA
