#include "FEMCommonKConsistent.h"

namespace {
template <typename Real> struct spectra_params;

template <> struct spectra_params<double> {
  size_t maxits{1000};
  double tol{1e-10};
  //-> This parameter must satisfy nev<ncv≤n, and is advised to take
  // ncv≥2⋅nev.
  //-> if ncv is positive, then its actual value is nev + ncv.
  //-> if ncv is negative, then its actual value is (-ncv+1)*nev.
  int ncv{-5};
};

template <> struct spectra_params<float> {
  size_t maxits{1000};
  float tol{1e-7};
  //-> This parameter must satisfy nev<ncv≤n, and is advised to take
  // ncv≥2⋅nev.
  //-> if ncv is positive, then its actual value is nev + ncv.
  //-> if ncv is negative, then its actual value is (-ncv+1)*nev.
  int ncv{-5};
};

template <typename Real, typename Solver>
int solve_impl(Solver &slv, Eigen::Matrix<Real, -1, -1> *basis,
               Eigen::Matrix<Real, -1, 1> *lambda, Spectra::SortRule selection,
               Spectra::SortRule sorting, size_t maxits, Real tol) {
  slv.init();
  int ncov = slv.compute(selection, maxits, tol, sorting);
  if (slv.info() == Spectra::CompInfo::Successful) {
    if (basis != nullptr) {
      *basis = slv.eigenvectors();
    }
    if (lambda != nullptr) {
      *lambda = slv.eigenvalues();
    }
  }
  return ncov;
}

template <typename Real, int Options = Eigen::RowMajor>
int general_eigen_value_decomposition(
    const Eigen::SparseMatrix<Real, Options> &A,
    const Eigen::SparseMatrix<Real, Eigen::ColMajor> &B, int num,
    Eigen::Matrix<Real, -1, -1> *basis = nullptr,
    Eigen::Matrix<Real, -1, 1> *lambda = nullptr,
    const Spectra::SortRule &rule = Spectra::SortRule::SmallestMagn,
    const spectra_params<Real> &prm = {}) {

  if (A.rows() != A.cols() || B.rows() != B.cols() || A.cols() != B.cols()) {
    std::cout << "failed!" << std::endl;
    std::cout << A.rows() << " " << A.cols() << " " << B.rows() << " " << B.cols() << std::endl;
    return -1;
  }

  int nev = std::min<int>(A.rows() - 1, num);
  int ncv = std::min<int>(A.rows(),
                          prm.ncv > 0 ? nev + prm.ncv : (-prm.ncv + 1) * nev);
  if (rule >= Spectra::SortRule::SmallestMagn &&
      rule <= Spectra::SortRule::SmallestAlge) {
    try {
      using OpType = Spectra::SymShiftInvert<Real, Eigen::Sparse, Eigen::Sparse,
                                             Eigen::Lower, Eigen::Lower,
                                             Options, Eigen::ColMajor>;
      using BOpType =
          Spectra::SparseSymMatProd<Real, Eigen::Lower, Eigen::ColMajor>;
      OpType op(A, B);
      BOpType Bop(B);
      Spectra::SymGEigsShiftSolver<OpType, BOpType,
                                   Spectra::GEigsMode::ShiftInvert>
          slv(op, Bop, nev, ncv, 0);
      int ret = solve_impl(slv, basis, lambda, (Spectra::SortRule)((int)rule - 4),
                        rule, prm.maxits, prm.tol);
      std::cout << "gevd num: "  << ret << std::endl;
      return ret;                        
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << ", use non-shift version.";
    }
  }

  using OpType = Spectra::SparseSymMatProd<Real, Eigen::Lower, Options>;
  using BOpType = Spectra::SparseCholesky<Real, Eigen::Lower, Eigen::ColMajor>;
  OpType op(A);
  BOpType Bop(B);
  Spectra::SymGEigsSolver<OpType, BOpType, Spectra::GEigsMode::Cholesky> slv(
      op, Bop, nev, ncv);
  int ret = solve_impl(slv, basis, lambda, rule, rule, prm.maxits, prm.tol);
  std::cout << "gevd num: "  << ret << std::endl;
  return ret;
}
}


namespace PhysIKA {
template <typename Real, int Options = Eigen::RowMajor>
Real k_consistent(const Eigen::SparseMatrix<Real, Options> &Kref,
                  const Eigen::Matrix<Real, -1, 1> &Mref,
                  const Eigen::SparseMatrix<Real, Options> &Kdest,
                  const Eigen::Matrix<Real, -1, 1> &Mdest, int start, int num) {
  Eigen::Matrix<Real, -1, 1> Lref, Ldest;
  general_eigen_value_decomposition<Real, Options>(Kref, (Eigen::SparseMatrix<Real, Eigen::ColMajor>)Mref.asDiagonal(), num+start, nullptr, &Lref);
  general_eigen_value_decomposition<Real, Options>(Kdest, (Eigen::SparseMatrix<Real, Eigen::ColMajor>)Mdest.asDiagonal(), num+start, nullptr, &Ldest);
  std::cout << "LRef: " << Lref.transpose() << std::endl;
  std::cout << "LDest: " << Ldest.transpose() << std::endl;
  return Lref.tail(Lref.size() - start).sum() / Ldest.tail(Ldest.size() - start).sum();
}

//-> Explicit Instantiation
template float k_consistent(const Eigen::SparseMatrix<float, Eigen::RowMajor> &Kref,
                  const Eigen::Matrix<float, -1, 1> &Mref,
                  const Eigen::SparseMatrix<float, Eigen::RowMajor> &Kdest,
                  const Eigen::Matrix<float, -1, 1> &Mdest, int start, int num);
template double k_consistent(const Eigen::SparseMatrix<double, Eigen::RowMajor> &Kref,
                  const Eigen::Matrix<double, -1, 1> &Mref,
                  const Eigen::SparseMatrix<double, Eigen::RowMajor> &Kdest,
                  const Eigen::Matrix<double, -1, 1> &Mdest,  int start,int num);     
template float k_consistent(const Eigen::SparseMatrix<float, Eigen::ColMajor> &Kref,
                  const Eigen::Matrix<float, -1, 1> &Mref,
                  const Eigen::SparseMatrix<float, Eigen::ColMajor> &Kdest,
                  const Eigen::Matrix<float, -1, 1> &Mdest, int start, int num);
template double k_consistent(const Eigen::SparseMatrix<double, Eigen::ColMajor> &Kref,
                  const Eigen::Matrix<double, -1, 1> &Mref,
                  const Eigen::SparseMatrix<double, Eigen::ColMajor> &Kdest,
                  const Eigen::Matrix<double, -1, 1> &Mdest, int start, int num);                               
}
