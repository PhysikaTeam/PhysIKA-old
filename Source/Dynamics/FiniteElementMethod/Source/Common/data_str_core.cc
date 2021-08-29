/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: data stream core helper.
 * @version    : 1.0
 */
#include "data_str_core.h"
#include <iostream>
using namespace std;
using namespace Eigen;
namespace PhysIKA {

template <typename T, size_t dim_>
dat_str_core<T, dim_>::dat_str_core(const size_t& dof, const bool hes_is_const)
    : dof_(dof), val_(0), gra_(dof * dim_), hes_(dof * dim_, dof * dim_), all_one_(Matrix<T, Dynamic, 1>::Ones(dof * dim_)), hes_is_const_(hes_is_const), whole_dim_(dim_ * dof)
{
    set_zero();
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::set_zero()
{
    val_ = 0;
    gra_.setZero();
    trips.resize(0);
    if (has_pre_compute_hes_)
    {
        if (!hes_is_const_)
            this->set_hes_zero_after_pre_compute();
    }
    else
        hes_.setZero();
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::hes_reserve(const VectorXi& nnzs)
{
    hes_.reserve(nnzs);
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::hes_compress()
{
    hes_.makeCompressed();
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::hes_add_diag(const size_t& time)
{
    hes_ += (time * all_one_).asDiagonal();
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_val(const T& val)
{

#pragma omp atomic
    val_ += val;
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::setFromTriplets()
{
    hes_.setZero();
    hes_.setFromTriplets(trips.begin(), trips.end());
    cout << "setfrom triples" << endl;
    return 0;
}

//>>>>>>>>>>>>>>>>>>save gra<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_gra(const Eigen::Matrix<T, Eigen::Dynamic, 1>& gra)
{
    gra_ += gra;
    return 0;
}
template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_gra(const size_t& pos, const Eigen::Matrix<T, dim_, 1>& point_gra)
{
    for (size_t d = 0; d < dim_; ++d)
    {
#pragma omp atomic
        gra_(dim_ * pos + d) += point_gra(d);
    }
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_gra(const size_t& pos, const T& one_gra)
{
#pragma omp atomic
    gra_(pos) += one_gra;

    return 0;
}

//>>>>>>>>>>>>>>>>>>save gra<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//>>>>>>>>>>>>>>>>>>save hes<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
template <typename T, size_t dim_>
int dat_str_core<T, dim_>::set_hes_zero_after_pre_compute()
{
    has_pre_compute_hes_ = true;
    hes_.makeCompressed();
    if (hes_is_const_)
        return 0;

    for (int k = 0; k < hes_.outerSize(); ++k)
        for (typename decltype(hes_)::InnerIterator it(hes_, k); it; ++it)
        {
            it.valueRef() = 0;
            // size_t id = it.col() * whole_dim_ + it.row();
            // hes_ref_[id] = &(it.valueRef());
        }
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_hes(const size_t& m, const size_t& n, const Eigen::Matrix<T, dim_, dim_>& loc_hes)
{
    if (!has_pre_compute_hes_)
    {
#pragma omp critical
        {
            for (size_t col = 0; col < dim_; ++col)
            {
                for (size_t row = 0; row < dim_; ++row)
                {
                    if (loc_hes(row, col))
                    {
                        // hes_.coeffRef(m * dim_ + row, n * dim_ + col) += loc_hes(row, col);
                        trips.push_back(Triplet<T>(m * dim_ + row, n * dim_ + col, loc_hes(row, col)));
                    }
                }
            }
        }
    }
    else
    {
        if (!hes_is_const_)
        {
            for (size_t col = 0; col < dim_; ++col)
            {
                for (size_t row = 0; row < dim_; ++row)
                {
                    if (loc_hes(row, col))
                    {
#pragma omp atomic
                        hes_.coeffRef(m * dim_ + row, m * dim_ + col) += loc_hes(row, col);
                    }
                }
            }
        }
    }
    return 0;
}

template <typename T, size_t dim_>
int dat_str_core<T, dim_>::save_hes(const size_t& row, const size_t& col, const T& value)
{
    if (!has_pre_compute_hes_)
    {
#pragma omp critical
        {
            trips.push_back(Triplet<T>(row, col, value));
        }
    }
    else
    {
        if (!hes_is_const_)
        {
// const size_t id = col * whole_dim_ + row;
// T& val = *hes_ref_[id];
#pragma omp atomic
            // val += value;
            hes_.coeffRef(row, col) += value;
        }
    }

    return 0;
}
//>>>>>>>>>>>>>>>>>>save hes<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

template <typename T, size_t dim_>
const T dat_str_core<T, dim_>::get_val() const
{
    return val_;
}

template <typename T, size_t dim_>
const Eigen::Matrix<T, Eigen::Dynamic, 1>& dat_str_core<T, dim_>::get_gra() const
{
    return gra_;
}

template <typename T, size_t dim_>
const typename dat_str_core<T, dim_>::SMP_TYPE& dat_str_core<T, dim_>::get_hes() const
{
    return hes_;
}

template <typename T, size_t dim_>
const size_t dat_str_core<T, dim_>::get_dof() const
{
    return dof_;
}

template class dat_str_core<double, 3>;
template class dat_str_core<double, 2>;
template class dat_str_core<double, 1>;
template class dat_str_core<float, 3>;
template class dat_str_core<float, 2>;
template class dat_str_core<float, 1>;
}  // namespace PhysIKA
