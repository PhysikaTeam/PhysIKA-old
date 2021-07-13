/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: basis function for finite element method.
 * @version    : 1.0
 */

#ifndef FEM_BASIS
#define FEM_BASIS
#include <Eigen/Dense>
#include <iostream>

namespace PhysIKA {
/**
 * shape function definition for finite element.
 *
 * Sample usage: 
 * auto v = shape_func::calc_basis_value(PNT, X);
 * shape_func::calc_Dhpi_Dxi(PNT, X, Dphi_Dxi);
 * double v = shape_func::volume();
 */
template <typename T, size_t dim_, size_t order_, size_t num_per_cell_>
struct shape_func
{
    static Eigen::Matrix<T, num_per_cell_, 1> calc_basis_value(const Eigen::Matrix<T, dim_, 1>& PNT, const T* X)
    {
        assert(0);
        return Eigen::Matrix<T, num_per_cell_, 1>::Zero();
    }
    static void calc_Dhpi_Dxi(const Eigen::Matrix<T, dim_, 1>& PNT, const T* X, Eigen::Matrix<T, num_per_cell_, dim_>& Dphi_Dxi)
    {
        std::cout << "unsupported type of basis func.\n";
        assert(0);
        return;
    }
    static double volume()
    {
        assert(0);
        return -1;
    }
};

/**
 * shape function definition for finite element.
 *
 * Sample usage: 
 * shape_func::calc_Dhpi_Dxi(PNT, X, Dphi_Dxi);
 * double v = shape_func::volume();
 */
template <typename T>
struct shape_func<T, 3, 1, 4>
{
    static void calc_Dphi_Dxi(const Eigen::Matrix<T, 3, 1>& PNT, const T* X, Eigen::Matrix<T, 4, 3>& Dphi_Dxi)
    {
        Dphi_Dxi.setZero();
        Dphi_Dxi.template topRows<3>().setIdentity();
        Dphi_Dxi.row(3) = Eigen::Matrix<T, 1, 3>::Ones() * (-1);
        return;
    }
    static double volume()
    {
        return 1.0 / 6;
    }
};

/**
 * shape function definition for finite element.
 *
 * Sample usage: 
 * auto v = shape_func::calc_basis_value(PNT, X);
 * shape_func::calc_Dhpi_Dxi(PNT, X, Dphi_Dxi);
 * double v = shape_func::volume();
 */
template <typename T>
struct shape_func<T, 3, 1, 8>
{
    static double volume()
    {
        return 8.0;
    }
    static Eigen::Matrix<T, 8, 1> calc_basis_value(const Eigen::Matrix<T, 3, 1>& PNT, const T* X)
    {
        Eigen::Matrix<T, 8, 1> basis_value = Eigen::Matrix<T, 8, 1>::Zero();
        const T                xi0 = PNT(0), xi1 = PNT(1), xi2 = PNT(2);
        // vector<T> l(3, 0);
        // vector<T> sign(3, 0);
        T l[3];
        T sign[3];
        for (size_t z = 0; z < 2; ++z)
        {
            sign[2] = z == 0 ? -1 : 1;
            l[2]    = 1 + sign[2] * xi2;
            for (size_t y = 0; y < 2; ++y)
            {
                sign[1] = y == 0 ? -1 : 1;
                l[1]    = 1 + sign[1] * xi1;
                for (size_t x = 0; x < 2; ++x)
                {
                    sign[0]           = x == 0 ? -1 : 1;
                    l[0]              = 1 + sign[0] * xi0;
                    const size_t p_id = z * 4 + y * 2 + (y == 0 ? x : 1 - x);
                    basis_value(p_id) = l[0] * l[1] * l[2];
                }
            }
        }
        basis_value /= 8.0;
        return basis_value;
    }

    static void calc_Dphi_Dxi(const Eigen::Matrix<T, 3, 1>& PNT, const T* X, Eigen::Matrix<T, 8, 3>& Dphi_Dxi)
    {
        Dphi_Dxi.setZero();
        const T xi0 = PNT(0), xi1 = PNT(1), xi2 = PNT(2);

        T l[3];
        T sign[3];
        for (size_t z = 0; z < 2; ++z)
        {
            sign[2] = z == 0 ? -1 : 1;
            l[2]    = 1 + sign[2] * xi2;
            for (size_t y = 0; y < 2; ++y)
            {
                sign[1] = y == 0 ? -1 : 1;
                l[1]    = 1 + sign[1] * xi1;
                for (size_t x = 0; x < 2; ++x)
                {
                    sign[0]           = x == 0 ? -1 : 1;
                    l[0]              = 1 + sign[0] * xi0;
                    const size_t p_id = z * 4 + y * 2 + (y == 0 ? x : 1 - x);
                    for (size_t d = 0; d < 3; ++d)
                        Dphi_Dxi(p_id, d) = sign[d] * l[(d + 1) % 3] * l[(d + 2) % 3];
                }
            }
        }
        Dphi_Dxi /= 8.0;
        return;
    }
};

/**
 * basis function definition for finite element.
 *
 * Sample usage: 
 */
template <typename T, size_t dim_, size_t field_, size_t order_, size_t num_per_cell_>

class basis_func
{
public:
    static void calc_Dphi_Dxi(const Eigen::Matrix<T, dim_, 1>& PNT, const T* X, Eigen::Matrix<T, num_per_cell_, dim_>& Dphi_Dxi)
    {
        return shape_func<T, dim_, order_, num_per_cell_>::calc_Dphi_Dxi(PNT, X, Dphi_Dxi);
    }

    static void calc_InvDm_Det(const Eigen::Matrix<T, num_per_cell_, dim_>& Dphi_Dxi, const T* X, T& Jac_det, Eigen::Matrix<T, dim_, dim_>& Dm_inv)
    {
        Dm_inv.setZero();
        const Eigen::Map<const Eigen::Matrix<T, dim_, num_per_cell_>> rest(X);
        Eigen::Matrix<T, dim_, dim_>                                  Dm = rest * Dphi_Dxi;
        Dm_inv                                                           = Dm.inverse();
        Jac_det                                                          = fabs(Dm.determinant()) * shape_func<T, dim_, order_, num_per_cell_>::volume();
        return;
    }
    static void get_def_gra(const Eigen::Matrix<T, num_per_cell_, dim_>& Dphi_Dxi, const T* const x, const Eigen::Matrix<T, dim_, dim_>& Dm_inv, Eigen::Matrix<T, field_, dim_>& def_gra)
    {
        const Eigen::Map<const Eigen::Matrix<T, field_, num_per_cell_>> deformed(x);
        def_gra = deformed * Dphi_Dxi * Dm_inv;
        return;
    }

    static void get_Ddef_Dx(const Eigen::Matrix<T, num_per_cell_, dim_>& Dphi_Dxi, const Eigen::Matrix<T, dim_, dim_>& Dm_inv, Eigen::Matrix<T, field_ * dim_, field_ * num_per_cell_>& Ddef_Dx)
    {
        Ddef_Dx.setZero();
        const Eigen::Matrix<T, num_per_cell_, dim_> Ddef_Dx_compressed = Dphi_Dxi * Dm_inv;
#pragma omp parallel for
        for (size_t i = 0; i < num_per_cell_; ++i)
            for (size_t j = 0; j < dim_; ++j)
                Ddef_Dx.block(j * field_, i * field_, field_, field_) = Eigen::Matrix<T, field_, field_>::Identity() * Ddef_Dx_compressed(i, j);
        return;
    }
};

}  // namespace PhysIKA
#endif
