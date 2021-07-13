/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: fast searching some partial eigenvalues
 * @version    : 1.0
 */
#include <iostream>

#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>

#include <Spectra/GenEigsRealShiftSolver.h>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>

#include "Common/config.h"

#include "search_eigenvalues.h"

using namespace std;
using namespace Eigen;
using namespace Spectra;

namespace PhysIKA {
/**
 * customize shift solver for spectra, more effective to compute some eigenvalues.
 *
 */
template <typename T>
class MyShitSolve
{
private:
    T              sigma_;
    const SPM_R<T> A_;
    const SPM_R<T> I_;
    SPM_R<T>       A_shift_;

public:
    MyShitSolve(const SPM_R<T>& A)
        : A_(A), I_(MAT<T>::Identity(A.rows(), A.cols()).sparseView()) {}
    int rows()
    {
        return A_.rows();
    }
    int cols()
    {
        return A_.cols();
    }
    void set_shift(T sigma)
    {
        sigma_   = sigma;
        A_shift_ = A_ - sigma_ * I_;
    }
    void perform_op(const T* x_in, T* y_out)
    {
        ConjugateGradient<SPM_R<T>, Lower | Upper, DiagonalPreconditioner<T>> cg;
        cg.compute(A_shift_);
        Map<const VEC<T>> IN(x_in, A_.cols());
        Map<VEC<T>>       OUT(y_out, A_.rows());
        OUT = cg.solve(IN);
    }
};

template <typename T>
int get_spectrum(const SPM_R<T>& A, const size_t band, const size_t num_band, MAT<T>& eig_vec, VEC<T>& eig_val)
{
    if (band == 0 || num_band <= 1)
        return 0;

    const size_t dim = A.rows(), num_eig = band * num_band;
    if (num_eig > dim)
    {
        cout << "band size * num_band is larger than the matrix size.\n";
        return __LINE__;
    }

    eig_vec = MAT<T>::Zero(dim, num_eig);
    eig_val = VEC<T>::Zero(num_eig);

    auto solve_fail_info = [](const int nconv, const int info) -> int {
        if (info == NOT_COMPUTED)
            printf("Spectra eigen solver not computed.Users should call the compute() member function of solvers.\n");
        else if (info == NOT_CONVERGING)
            printf("Spectra eigen solver not converge. The number of converged eigenvalues are %d\n", nconv);
        else
            printf("Spectra eigen solver: Used in Cholesky decomposition, indicating that the matrix is not positive definite.\n");
        return __LINE__;
    };

    Map<const SPM_C<T>>    transA(A.rows(), A.cols(), A.nonZeros(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), 0);
    SparseSymShiftSolve<T> shift_op(transA);
    // MyShitSolve<T> shift_op(A);
    T            max_eig_value, min_eig_value, interval;
    const size_t n = band * 3 >= A.rows() ? A.rows() - 1 : band * 3;
    {  //solve max eigvalue
        SparseSymMatProd<T>                                 op(transA);
        SymEigsSolver<T, LARGEST_MAGN, SparseSymMatProd<T>> eig_solver(&op, band, n);
        eig_solver.init();
        int nconv = eig_solver.compute();
        if (eig_solver.info() == SUCCESSFUL)
        {
            eig_vec.rightCols(band)  = eig_solver.eigenvectors().rowwise().reverse();
            eig_val.bottomRows(band) = eig_solver.eigenvalues().reverse();
        }
        else
            return solve_fail_info(nconv, eig_solver.info());
        max_eig_value = eig_val(num_eig - 1);
    }
    {  //solve min eigvalue
        SymEigsShiftSolver<T, LARGEST_MAGN, SparseSymShiftSolve<T>> eig_solver(&shift_op, band, n, 0.0);
        // SymEigsShiftSolver<T, LARGEST_MAGN, MyShitSolve<T>> eig_solver(&shift_op, band, n, 0.0);
        eig_solver.init();
        int nconv = eig_solver.compute();
        if (eig_solver.info() == SUCCESSFUL)
        {
            eig_vec.leftCols(band) = eig_solver.eigenvectors().rowwise().reverse();
            eig_val.topRows(band)  = eig_solver.eigenvalues().reverse();
        }
        else
            return solve_fail_info(nconv, eig_solver.info());
        min_eig_value = eig_val(0);
    }
    interval = (max_eig_value - min_eig_value) / (num_band - 1);

    {  //solve middle eigvalue
        for (size_t i = 0; i < num_band - 2; ++i)
        {
            T                                                           val = min_eig_value + interval * (i + 1);
            SymEigsShiftSolver<T, LARGEST_MAGN, SparseSymShiftSolve<T>> eig_solver(&shift_op, band, n, val);
            // SymEigsShiftSolver<T, LARGEST_MAGN, MyShitSolve<T>> eig_solver(&shift_op, band, n, val);
            eig_solver.init();
            int nconv = eig_solver.compute();
            if (eig_solver.info() == SUCCESSFUL)
            {
                eig_vec.middleCols((i + 1) * band, band) = eig_solver.eigenvectors().rowwise().reverse();
                eig_val.middleRows((i + 1) * band, band) = eig_solver.eigenvalues().reverse();
            }
            else
                return solve_fail_info(nconv, eig_solver.info());
        }
    }
    return 0;
}

template int get_spectrum(const SPM_R<double>& A, const size_t band, const size_t num_band, MAT<double>& eig_vec, VEC<double>& eig_val);
template int get_spectrum(const SPM_R<float>& A, const size_t band, const size_t num_band, MAT<float>& eig_vec, VEC<float>& eig_val);

template <typename T>
int eig(const SPM_R<T>& A, MAT<T>& eig_vec, VEC<T>& eig_val)
{
    const size_t                                        dim    = A.rows();
    SparseMatrix<T>                                     Atrans = A.transpose();
    SparseSymMatProd<T>                                 op(Atrans);
    SymEigsSolver<T, LARGEST_MAGN, SparseSymMatProd<T>> eig_solver(&op, dim / 10, dim / 5);
    eig_solver.init();
    int nconv = eig_solver.compute();
    if (eig_solver.info() == SUCCESSFUL)
    {
        eig_vec = eig_solver.eigenvectors();
        eig_val = eig_solver.eigenvalues();
        return 0;
    }
    else
    {
        if (eig_solver.info() == NOT_COMPUTED)
            printf("Spectra eigen solver not computed.Users should call the compute() member function of solvers.\n");
        else if (eig_solver.info() == NOT_CONVERGING)
            printf("Spectra eigen solver not converge. The number of converged eigenvalues are %d\n", nconv);
        else
            printf("Spectra eigen solver: Used in Cholesky decomposition, indicating that the matrix is not positive definite.\n");
        return __LINE__;
    }
}

template int eig(const SPM_R<double>& A, MAT<double>& eig_vec, VEC<double>& eig_val);
template int eig(const SPM_R<float>& A, MAT<float>& eig_vec, VEC<float>& eig_val);

double find_max_eigenvalue(const Eigen::SparseMatrix<double>& A, const size_t max_itrs)
{
    VectorXd v = VectorXd::Random(A.cols());
    // v = v / v.norm();

    // double last_lambda = 1e40, new_lambda = 0;
    // for(size_t i = 0; i < max_itrs; ++i){
    //   v = A * v;
    //   new_lambda = v.norm();
    //   if(fabs((new_lambda - last_lambda)) > 1e-20)
    //     last_lambda = new_lambda;
    //   else{
    //     v = v / new_lambda;
    //     break;
    //   }
    //   v = v / new_lambda;
    // }
    double   last_lambda = 1e40, new_lambda = 0;
    VectorXd w(v.size());
    for (size_t i = 0; i < max_itrs; ++i)
    {
        w          = A * v;
        new_lambda = v.dot(w);
        if (fabs(new_lambda - last_lambda) > 1e-20)
            last_lambda = new_lambda;
        else
            break;

        v.array() = w.array() - w.sum() / w.size();
        v.array() /= v.norm();
    }

    return new_lambda;
}

double find_min_eigenvalue(const Eigen::SparseMatrix<double>& A, const size_t max_itrs)
{
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(A);

    VectorXd v = VectorXd::Ones(A.cols());
    v          = v / v.norm();

    double last_v = 1e40, new_v = 0;
    for (size_t i = 0; i < max_itrs; ++i)
    {
        v     = cg.solve(v);
        new_v = v.norm();
        if (fabs(new_v - last_v) > 1e-20)
            last_v = new_v;
        else
            break;
        v = v / new_v;
    }
    return 1.0 / new_v;
}

double find_min_eigenvalue(const Eigen::SparseMatrix<double>& A, const double& max_eig, const size_t max_iters)
{
    SparseMatrix<double> B = A;
#pragma omp parallel for
    for (int k = 0; k < A.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it)
            if (it.index() == k)
            {
                it.valueRef() -= max_eig;
                break;
            }
    double min_eig = find_max_eigenvalue(B);
    min_eig        = fabs(max_eig) - fabs(min_eig);
    return min_eig;
}
double find_condition_number(const Eigen::SparseMatrix<double>& A, const size_t max_itrs)
{
    double
        max_eig = find_max_eigenvalue(A, max_itrs),
        min_eig = find_min_eigenvalue(A, max_eig, max_itrs);
    cout << "max eig value is " << max_eig << " min eig " << min_eig << endl;
    return max_eig / min_eig;
}

int find_max_min_eigenvalues(const Eigen::SparseMatrix<double>& A, double& max_eigvalue, double& min_eigvalue)
{
    SparseGenMatProd<double> op(A);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<double, LARGEST_MAGN, SparseGenMatProd<double>> eigs(&op, 1, 3);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXcd evalues;
    if (eigs.info() == SUCCESSFUL)
        evalues = eigs.eigenvalues();

    max_eigvalue = evalues(0).real();

    SparseGenRealShiftSolve<double> op_min(A);
    // Construct eigen solver object with shift 0
    // This will find eigenvalues that are closest to 0
    GenEigsRealShiftSolver<double, LARGEST_MAGN, SparseGenRealShiftSolve<double>> eigs_min(&op_min, 1, 3, 0.0);

    eigs_min.init();
    eigs_min.compute();

    if (eigs_min.info() == SUCCESSFUL)
    {
        Eigen::VectorXd evalues = eigs_min.eigenvalues().real();
        min_eigvalue            = evalues(0);
    }

    cout << "max eigvalue " << max_eigvalue << " min eigvalue " << min_eigvalue << endl;
    return 0;
}
#if 0
//this jacobian transform is used for large sparse matrix
template<typename T>
int eig_jac(const SPM_R<T>& mat_A, MAT<T>&eig_vec, VEC<T>& eig_val){
  const size_t dim = mat_A.rows();
  constexpr T pi = 3.1415926535897932384626433832;

  SPM_R<T> Q(dim, dim);{
    vector<Triplet<T>> trips(dim);
#pragma omp parallel for
    for(size_t i = 0; i < dim; ++i)
      trips[i] = Triplet<T>(i, i, 1.0);
    Q.reserve(dim + 2);
    Q.setFromTriplets(trips.begin(), trips.end());
  }

  auto get_max_off_diag = [](const SPM_R<T>& A, size_t& m, size_t& n, T& A_mn, T& A_mm, T& A_nn)->T{
    __TIME_BEGIN__;
    T max_value = 0;
#pragma omp parallel for
    for(size_t k = 0; k < A.outerSize(); ++k)
      for(typename SPM_R<T>::InnerIterator it(A, k); it; ++it){
        if(it.index() < k && fabs(it.value()) > max_value){
#pragma omp critical
          {
            if(fabs(it.value()) > max_value){
              m = it.col();
              n = it.row();
              max_value = fabs(it.value());
              A_mn = it.value();
            }
          }
        }
      }
    A_mm = A.coeff(m, m);
    A_nn = A.coeff(n, n);
    __TIME_END__("get max value");
    return max_value;
  };


  //Here we only change the lower triangle part of A.
  auto RTAR = [](const size_t& m, const size_t& n, const T& cos_theta, const T& sin_theta, SPM_R<T>&A)->void{
    __TIME_BEGIN__;
    //calculate B = RT * A
    SparseVector<T> row_m = A.row(m), row_n = A.row(n);
    SparseVector<T> new_row_m = cos_theta * row_m + sin_theta * row_n;
    SparseVector<T> new_row_n = -sin_theta * row_m + cos_theta * row_n;
    A.row(m) = new_row_m;
    A.row(n) = new_row_n;
    __TIME_END__("RT A");

    //calculate C = B * R
    //This transpose should be fast if Eigen library is coded well
    //TODO: confirm this by comparing the cost with just copying three pointers to construct BT.
    __TIME_BEGIN__;
    SparseMatrix<T, ColMajor> BT = A;

    //should also be very fast if Eigen library is coded well
    SparseVector<T, ColMajor> col_m = BT.col(m);
    SparseVector<T, ColMajor> col_n = BT.col(n);
    SparseVector<T, ColMajor> new_col_m = cos_theta * col_m + sin_theta * col_n;
    SparseVector<T, ColMajor> new_col_n = -sin_theta * col_m + cos_theta * col_n;

    BT.col(m) = new_col_m;
    BT.col(n) = new_col_n;
    A = BT;
    __TIME_END__("RT A R");
  };

  VEC<T> col_m(dim), col_n(dim);
  // multiply all the R
  auto PI_R = [&col_m, &col_n](
      const size_t& m, const size_t& n, const T& cos_theta, const T& sin_theta,  MAT<T>& V)->void{
    col_m = V.col(m);
    col_n = V.col(n);
    V.col(m).noalias() = cos_theta * col_m - sin_theta * col_n;
    V.col(n).noalias() = sin_theta * col_m + cos_theta * col_n;
  };

  T cos_theta, sin_theta, A_mn, A_mm, A_nn;
  size_t m, n;
  SPM_R<T> An  = mat_A;
  eig_vec = MAT<T>::Identity(dim, dim);




  while(fabs(get_max_off_diag(An, m, n, A_mn, A_mm, A_nn)) > 1e-8){
    printf("max off-diagonal element: %f.\n", A_mn);
    if(fabs(A_mm - A_nn) < 1e-8){
      T theta =  A_mn > 0 ? pi / 4 : -pi / 4;
      cos_theta = cos(theta); sin_theta = sin(theta);
    }
    else{
      T c = (A_mm - A_nn) / (2 * A_mn);
      T tan_theta = 1.0 / (sqrt(c * c + 1) + fabs(c));
      if(c < 0)
        tan_theta *= -1;
      cos_theta = 1.0 / sqrt(1 + tan_theta * tan_theta);
      sin_theta = cos_theta * tan_theta;
    }
    RTAR(m, n, cos_theta, sin_theta, An);
    PI_R(m, n, cos_theta, sin_theta, eig_vec);
    cout << endl;
  };
  eig_val = An.diagonal();
  return 0;
}
#endif

template <typename T>
int eig_jac(const mat3<T>& mat_A, mat3<T>& eig_vec, Eigen::Matrix<T, 3, 1>& eig_val)
{
    if (fabs((mat_A - mat_A.transpose()).maxCoeff()) > 1e-8)
    {
        std::cerr << "mat_A is not sysmetic." << std::endl;
        return __LINE__;
    }

    auto get_max_off_diag = [](const mat3<T>& A, size_t& m, size_t& n) -> T {
        T val_01 = fabs(A(0, 1)), val_02 = fabs(A(0, 2)), val_12 = fabs(A(1, 2));

        T value;
        if (val_01 >= val_02 && val_01 >= val_12)
        {
            m     = 0;
            n     = 1;
            value = A(0, 1);
        }
        else if (val_02 >= val_01 && val_02 >= val_12)
        {
            m     = 0;
            n     = 2;
            value = A(0, 2);
        }
        else if (val_12 >= val_01 && val_12 >= val_02)
        {
            m     = 1;
            n     = 2;
            value = A(1, 2);
        }
        return value;
    };

    auto get_Q = [](const size_t& m, const size_t& n, const T& val_mn) -> mat3<T> {
        mat3<T> Q     = mat3<T>::Identity();
        T       pi    = 3.1415926535897932384626433832;
        T       theta = val_mn > 0 ? pi / 4 : -pi / 4;
        Q(m, m) = Q(n, n) = cos(theta);
        Q(m, n)           = -sin(theta);
        Q(n, m)           = -Q(m, n);
        return Q;
    };
    auto get_Q_tan = [](const size_t& m, const size_t& n, const T& tan_theta) -> mat3<T> {
        mat3<T> Q         = mat3<T>::Identity();
        T       cos_theta = 1.0 / sqrt(1 + tan_theta * tan_theta);
        Q(m, m)           = cos_theta;
        Q(n, n)           = cos_theta;
        Q(m, n)           = -tan_theta * Q(m, m);
        Q(n, m)           = -Q(m, n);
        return Q;
    };

    eig_vec.setIdentity();

    size_t  m, n;
    mat3<T> An = mat_A;
    mat3<T> Q;
    while (fabs(get_max_off_diag(An, m, n)) > 1e-8)
    {
        if (fabs(An(m, m) - An(n, n)) < 1e-8)
            Q = get_Q(m, n, An(m, n));
        else
        {
            T c         = (An(m, m) - An(n, n)) / (2 * An(m, n));
            T tan_theta = 1.0 / (sqrt(c * c + 1) + fabs(c));
            if (c < 0)
                tan_theta *= -1;
            Q = get_Q_tan(m, n, tan_theta);
        }
        cout << "An " << endl
             << An << endl
             << " Q " << endl
             << Q << endl
             << An * Q << endl
             << Q.transpose() * An * Q << endl;
        ;
        An      = Q.transpose() * An * Q;
        eig_vec = eig_vec * Q;
    };
    eig_val = An.diagonal();

    // cout << "check \n";
    // cout << eig_vec.transpose() * eig_val.asDiagonal() * eig_vec << endl;

    return 0;
}

template int eig_jac<double>(const mat3<double>& mat_A, mat3<double>& eig_vec, Eigen::Matrix<double, 3, 1>& eig_val);
template int eig_jac<float>(const mat3<float>& mat_A, mat3<float>& eig_vec, Eigen::Matrix<float, 3, 1>& eig_val);

}  // namespace PhysIKA
