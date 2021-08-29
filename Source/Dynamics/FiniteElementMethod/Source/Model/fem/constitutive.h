/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: constitutive equation for finite element method.
 * @version    : 1.0
 */
#ifndef FEM_CONSTITUTIVE
#define FEM_CONSTITUTIVE

#include <cmath>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "Common/polar_decomposition.h"
#include "Common/tensor.h"

namespace PhysIKA {
/**
 * consitutive energy for finite element method.
 *
 */
template <typename T, size_t dim_, size_t field_>
class constitutive
{
public:
    static T
    val(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr);
    static Eigen::Matrix<T, field_ * dim_, 1>
    gra(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr);

    static Eigen::Matrix<T, field_ * dim_, field_ * dim_>
    hes(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr);
};

template <typename T, size_t dim_, size_t field_>
class quadratic_csttt : public constitutive<T, dim_, field_>
{
public:
    static T
    val(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        return 0.5 * mtr(0) * ((F.array() * F.array()).sum());
    }
    static Eigen::Matrix<T, field_ * dim_, 1>
    gra(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        Eigen::Map<const Eigen::Matrix<T, field_ * dim_, 1>> F_vec(F.data());
        Eigen::Matrix<T, field_ * dim_, 1>                   gra_vec = mtr(0) * F_vec;
        return gra_vec;
    }
    static Eigen::Matrix<T, field_ * dim_, field_ * dim_>
    hes(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        const static Eigen::Matrix<T, field_ * dim_, field_* dim_> hes =
            mtr(0) * Eigen::Matrix<T, field_ * dim_, field_ * dim_>::Identity();
        return std::move(hes);
    }
};

template <typename T, size_t dim_, size_t field_>
class elas_csttt : public constitutive<T, dim_, field_>
{
};

template <typename T, size_t dim_, size_t field_>
class arap_csttt : public constitutive<T, dim_, field_>
{

public:
    static T
    val(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        T                            lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> R;
        int                          res = polar_decomposition<T, Eigen::Matrix<T, dim_, dim_>, 3>(F, R);
        // error_msg_cond(res != 0, "polar_decomposition failed.");
        return lam / 2 * (F - R).squaredNorm();
    }

    static Eigen::Matrix<T, dim_ * dim_, 1>
    gra(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        T                            lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> R;
        int                          res = polar_decomposition<T, Eigen::Matrix<T, dim_, dim_>, 3>(F, R);
        // error_msg_cond(res != 0, "polar_decomposition failed.");
        Eigen::Matrix<T, dim_ * dim_, 1> gra_vec;
        Eigen::Map<Eigen::Matrix<T, dim_, dim_>>(gra_vec.data()) = lam * (F - R);
        return std::move(gra_vec);
    }

    static Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>
    hes(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        T                                          lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_ * dim_, dim_ * dim_> hes;
        hes.setIdentity();
        hes *= lam;
        return std::move(hes);
    }
};

template <typename T, size_t dim_, size_t field_>
class linear_csttt : public elas_csttt<T, dim_, field_>
{
public:
    static T
    val(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        const T                      lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> strain =
            0.5 * (F + F.transpose()) - Eigen::Matrix<T, dim_, dim_>::Identity();
        return mu * (strain.array() * strain.array()).sum() + 0.5 * lam * strain.trace() * strain.trace();
    }

    static Eigen::Matrix<T, dim_ * dim_, 1>
    gra(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        const T                            lam = mtr(0), mu = mtr(1);
        const Eigen::Matrix<T, dim_, dim_> Iden   = Eigen::Matrix<T, dim_, dim_>::Identity();
        Eigen::Matrix<T, dim_, dim_>       strain = 0.5 * (F + F.transpose()) - Iden;
        Eigen::Matrix<T, dim_, dim_>       gra_mat =
            mu * (F + F.transpose() - 2 * Iden) + lam * (F - Iden).trace() * Iden;
        Eigen::Map<Eigen::Matrix<T, dim_ * dim_, 1>> gra_vec(gra_mat.data());
        return std::move(gra_vec);
    }

    static Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>
    hes(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        const T                                           lam = mtr(0), mu = mtr(1);
        static Eigen::Matrix<T, dim_ * dim_, dim_ * dim_> hes;
        static bool                                       have_calc = false;
        if (!have_calc)
        {
            const Eigen::Matrix<T, dim_ * dim_, dim_* dim_> Iden =
                Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>::Identity();
            Eigen::Matrix<T, dim_ * dim_, dim_* dim_> DDtrace =
                Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>::Zero();
            for (size_t row = 0; row < dim_ * dim_; row += dim_ + 1)
            {
                for (size_t col = 0; col < dim_ * dim_; col += dim_ + 1)
                {
                    DDtrace(row, col) = 1;
                }
            }

            Eigen::Matrix<T, dim_ * dim_, dim_* dim_> Dsquare =
                Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>::Zero();
            {
                for (size_t row = 0; row < dim_; ++row)
                {
                    for (size_t col = 0; col < dim_; ++col)
                    {
                        Dsquare(row * dim_ + col, col * dim_ + row) = 1;
                    }
                }
                Dsquare += Iden;
            }
            hes       = mu * Dsquare + lam * DDtrace;
            have_calc = true;
        }

        return std::move(hes);
    }
};

template <typename T, size_t dim_, size_t field_>
class stvk : public elas_csttt<T, dim_, field_>
{
public:
    using tensor_type = fourth_tensor<T, dim_, dim_, dim_, dim_>;
    static T
    val(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        const T                      lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> strain =
            0.5 * (F.transpose() * F - Eigen::Matrix<T, dim_, dim_>::Identity());
        return mu * (strain.array() * strain.array()).sum() + 0.5 * lam * strain.trace() * strain.trace();
    }

    static Eigen::Matrix<T, dim_ * dim_, 1>
    gra(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        T                                  lam = mtr(0), mu = mtr(1);
        const Eigen::Matrix<T, dim_, dim_> Iden   = Eigen::Matrix<T, dim_, dim_>::Identity();
        Eigen::Matrix<T, dim_, dim_>       strain = 0.5 * (F.transpose() * F - Iden);
        Eigen::Matrix<T, dim_, dim_>       gra_mat =
            F * (2 * mu * strain + lam * strain.trace() * Iden);
        Eigen::Map<Eigen::Matrix<T, dim_ * dim_, 1>> gra_vec(gra_mat.data());
        return std::move(gra_vec);
    }

    static Eigen::Matrix<T, dim_ * dim_, dim_ * dim_>
    hes(const Eigen::Matrix<T, dim_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&      mtr)
    {
        T                            lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> strain =
            0.5 * (F.transpose() * F - Eigen::Matrix<T, dim_, dim_>::Identity());
        const Eigen::Matrix<T, dim_, dim_>                Iden = Eigen::Matrix<T, dim_, dim_>::Identity();
        static Eigen::Matrix<T, dim_ * dim_, dim_ * dim_> hes;
        // TODO: fill this hes
        tensor_type dF_dF;
        {
            for (size_t row_out = 0; row_out < dim_; ++row_out)
            {
                for (size_t col_out = 0; col_out < dim_; ++col_out)
                {
                    Eigen::Matrix<T, dim_, dim_> zero = Eigen::Matrix<T, dim_, dim_>::Zero();
                    zero(row_out, col_out)            = 1;
                    dF_dF(row_out, col_out)           = zero;
                }
            }
        }

        Eigen::Matrix<T, dim_, dim_> rhs = 2 * mu * strain + lam * strain.trace() * Iden;

        tensor_type drhs_dF;
        {
            decltype(dF_dF) dstrain_dF;
            {
                for (size_t row_out = 0; row_out < dim_; ++row_out)
                {
                    for (size_t col_out = 0; col_out < dim_; ++col_out)
                    {
                        Eigen::Matrix<T, dim_, dim_> zero = Eigen::Matrix<T, dim_, dim_>::Zero();
                        zero.col(row_out) += F.col(col_out);
                        zero.col(col_out) += F.col(row_out);
                        dstrain_dF(row_out, col_out) = mu * zero;
                    }
                }
            }

            tensor_type dtrace_dF;
            {
                for (size_t row_out = 0; row_out < dim_; ++row_out)
                {
                    dtrace_dF(row_out, row_out) = lam * F;
                }
            }
            drhs_dF = dstrain_dF + dtrace_dF;
        }

        tensor_type hes_tensor = dF_dF * rhs + F * drhs_dF;
        hes_tensor.Flatten(hes);

        return std::move(hes);
    }
};

template <typename T, size_t dim_, size_t field_>
class corotated_csttt : public constitutive<T, dim_, field_>
{
public:
    static T
    val(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        const T                                          lam = mtr(0), mu = mtr(1);
        Eigen::JacobiSVD<Eigen::Matrix<T, field_, dim_>> svd(
            F, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // res is Sigma - I, where Sigma is the singularValues of F.
        // Sigma is a diagonal matrix.
        auto res = svd.singularValues();
        for (int i = 0; i < res.size(); ++i)
        {
            res[i] -= 1;
        }
        // mu * ||Sigma - I||_F + lambda/2 * tr^2(Sigma - I).
        return mu * res.dot(res) + 0.5 * lam * res.sum() * res.sum();
    }
    static Eigen::Matrix<T, field_ * dim_, 1>
    gra(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        const T                      lam = mtr(0), mu = mtr(1);
        Eigen::Matrix<T, dim_, dim_> R;
        {
            int res = polar_decomposition<T, Eigen::Matrix<T, dim_, dim_>, 3>(F, R);
            // error_msg_cond(res != 0, "polar_decomposition failed.");
        }
        Eigen::JacobiSVD<Eigen::Matrix<T, field_, dim_>> svd(
            F, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // res is Sigma - I, where Sigma is the singularValues of F.
        // Sigma is a diagonal matrix.
        const auto& res = svd.singularValues();
        // 2*mu*(F-R) + lambda * tr(Sigma - I) * R
        Eigen::Matrix<T, dim_, dim_> gra_mat =
            2 * mu * (F - R) + lam * (res.sum() - 3) * R;
        Eigen::Map<Eigen::Matrix<T, dim_ * dim_, 1>> gra_vec(gra_mat.data());
        return std::move(gra_vec);
    }

    static Eigen::Matrix<T, field_ * dim_, field_ * dim_>
    hes(const Eigen::Matrix<T, field_, dim_>& F,
        const Eigen::Matrix<T, -1, 1>&        mtr)
    {
        const T lam = mtr(0), mu = mtr(1);
        // 2*mu*Iden + lambda * d(tr(R^TF)*R)
        Eigen::Matrix<T, dim_ * dim_, dim_ * dim_> hes;
        // part 1. 2*mu*Iden
        hes.setIdentity();
        hes *= 2 * mu;
        // part 2. lambda * d(tr(R^TF)*R)
        Eigen::Matrix<T, dim_, dim_> R;
        {
            int res = polar_decomposition<T, Eigen::Matrix<T, dim_, dim_>, 3>(F, R);
            // error_msg_cond(res != 0, "polar_decomposition failed.");
        }
        // convert the R matrix to a vector. (column major)
        Eigen::Map<const Eigen::Matrix<T, dim_ * dim_, 1>> R_vec(
            R.data(), dim_ * dim_, 1);
        hes += lam * R_vec * R_vec.transpose();
        return std::move(hes);
    }
};

}  // namespace PhysIKA
#endif
