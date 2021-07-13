/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: eigenvalues for sparse matrix jacobian
 * @version    : 1.0
 */
#include "Common/config.h"
#include "spm_eig_jac.h"
using namespace std;
using namespace Eigen;
namespace PhysIKA {

template <typename T>
T get_max_off_diag(const SPM_R<T>& A, size_t& m, size_t& n, T& A_mn, T& A_mm, T& A_nn)
{
    T max_value = 0;
#pragma omp parallel for
    for (size_t k = 0; k < A.outerSize(); ++k)
        for (typename SPM_R<T>::InnerIterator it(A, k); it; ++it)
        {
            if (it.index() < k && fabs(it.value()) > max_value)
            {
#pragma omp critical
                {
                    if (fabs(it.value()) > max_value)
                    {
                        m         = it.col();
                        n         = it.row();
                        max_value = fabs(it.value());
                        A_mn      = it.value();
                    }
                }
            }
        }
    A_mm = A.coeff(m, m);
    A_nn = A.coeff(n, n);
    return max_value;
}

template <typename T>
int jac_rotate(const size_t& m, const size_t& n, const T& cos_theta, const T& sin_theta, SPM_R<T>& A)
{
    __TIME_BEGIN__;
    //calculate B = RT * A
    SparseVector<T> row_m = A.row(m), row_n = A.row(n);
    SparseVector<T> new_row_m = cos_theta * row_m + sin_theta * row_n;
    SparseVector<T> new_row_n = -sin_theta * row_m + cos_theta * row_n;
    new_row_m.pruned(1e-5);
    new_row_n.pruned(1e-5);
    A.row(m) = new_row_m;
    A.row(n) = new_row_n;
    __TIME_END__("RT A", false);

    //calculate C = B * R
    //This transpose should be fast if Eigen library is coded well
    //TODO: confirm this by comparing the cost with just copying three pointers to construct BT.
    __TIME_BEGIN__;
    SparseMatrix<T, ColMajor> BT = A;

    //should also be very fast if Eigen library is coded well
    SparseVector<T, ColMajor> col_m     = BT.col(m);
    SparseVector<T, ColMajor> col_n     = BT.col(n);
    SparseVector<T, ColMajor> new_col_m = cos_theta * col_m + sin_theta * col_n;
    SparseVector<T, ColMajor> new_col_n = -sin_theta * col_m + cos_theta * col_n;
    new_col_m.pruned(1e-5);
    new_col_n.pruned(1e-5);
    BT.col(m) = new_col_m;
    BT.col(n) = new_col_n;
    A         = BT;
    __TIME_END__("RT A R", false);
    return 0;
}

template <typename T>
int eig_jac(const SPM_R<T>& mat_A, MAT<T>& eig_vec, VEC<T>& eig_val)
{
    const size_t dim = mat_A.rows();
    constexpr T  pi  = 3.1415926535897932384626433832;

    SPM_C<T>           Q(dim, dim);
    vector<Triplet<T>> trips(dim);
    {
#pragma omp parallel for
        for (size_t i = 0; i < dim; ++i)
            trips[i] = Triplet<T>(i, i, 1.0);
        Q.reserve(dim);
        Q.setFromTriplets(trips.begin(), trips.end());
    }

    auto gen_R = [&Q, &trips](const size_t& m, const size_t& n, const T& cos_theta, const T& sin_theta) -> void {
        Q.setFromTriplets(trips.begin(), trips.end());
        Q.coeffRef(m, m) = cos_theta;
        Q.coeffRef(n, n) = cos_theta;
        Q.coeffRef(m, n) = -sin_theta;
        Q.coeffRef(n, m) = sin_theta;
        Q.makeCompressed();
    };

    VEC<T> col_m(dim), col_n(dim);
    // multiply all the R
    auto PI_R = [&col_m, &col_n](
                    const size_t& m, const size_t& n, const T& cos_theta, const T& sin_theta, MAT<T>& V) -> void {
        col_m              = V.col(m);
        col_n              = V.col(n);
        V.col(m).noalias() = cos_theta * col_m + sin_theta * col_n;
        V.col(n).noalias() = -sin_theta * col_m + cos_theta * col_n;
    };

    T        cos_theta, sin_theta, A_mn, A_mm, A_nn;
    size_t   m, n;
    SPM_R<T> An = mat_A;
    eig_vec     = MAT<T>::Identity(dim, dim);
    size_t cnt  = 0;
    while (fabs(get_max_off_diag(An, m, n, A_mn, A_mm, A_nn)) > 1e-8)
    {
        if (cnt++ % 100 == 0)
            printf("max off-diagonal element: %f and non Zeros is %ld.\n", A_mn, An.nonZeros());

        if (fabs(A_mm - A_nn) < 1e-8)
        {
            T theta   = A_mn > 0 ? pi / 4 : -pi / 4;
            cos_theta = cos(theta);
            sin_theta = sin(theta);
        }
        else
        {
            T c         = (A_mm - A_nn) / (2 * A_mn);
            T tan_theta = 1.0 / (sqrt(c * c + 1) + fabs(c));
            if (c < 0)
                tan_theta *= -1;
            cos_theta = 1.0 / sqrt(1 + tan_theta * tan_theta);
            sin_theta = cos_theta * tan_theta;
        }
        // RTAR(m, n, cos_theta, sin_theta, An);
        gen_R(m, n, cos_theta, sin_theta);
        // {
        //   __TIME_BEGIN__;
        //   SPM_C<T> B = (An * Q).pruned(1e-8);
        //   __TIME_END__("AR");
        //   //This equals to transpose
        //   Map<const SPM_R<T>> trans_Q(Q.rows(), Q.cols(), Q.nonZeros(), Q.outerIndexPtr(), Q.innerIndexPtr(), Q.valuePtr(), 0);
        //   __TIME_BEGIN__;
        //   An = (trans_Q * B).pruned(1e-8);
        //   __TIME_END__("RTAR");
        // }
        jac_rotate(m, n, cos_theta, sin_theta, An);
        PI_R(m, n, cos_theta, sin_theta, eig_vec);
    };
    eig_val = An.diagonal();
    return 0;
}

template int eig_jac<double>(const SPM_R<double>& mat_A, MAT<double>& eig_vec, VEC<double>& eig_val);
template int eig_jac<float>(const SPM_R<float>& mat_A, MAT<float>& eig_vec, VEC<float>& eig_val);
}  // namespace PhysIKA
