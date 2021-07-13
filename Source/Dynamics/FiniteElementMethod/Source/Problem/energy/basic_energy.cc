/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: some basic energy function.
 * @version    : 1.0
 */
#include <iostream>

#include <Eigen/SparseCore>

#include "basic_energy.h"

using namespace std;
using namespace Eigen;

namespace PhysIKA {
/******************************************momentum*******************************/

template <typename T, size_t dim_>
size_t momentum<T, dim_>::Nx() const
{
    return dim_ * dof_;
}

template <typename T, size_t dim_>
momentum<T, dim_>::momentum(const size_t dof, const Matrix<T, -1, 1>& mass_vec, const T& dt)
    : dof_(dof), dispk_(Matrix<T, -1, 1>::Zero(dim_ * dof)), vk_(Matrix<T, -1, 1>::Zero(dim_ * dof)), dt_(dt), d1dt_(1 / dt), d1dtdt_(1 / dt / dt), mass_vec_(dim_ * dof)
{

#pragma omp parallel for
    for (size_t i = 0; i < dof; ++i)
    {
        for (size_t j = 0; j < dim_; ++j)
        {
            mass_vec_(i * dim_ + j) = mass_vec(i);
        }
    }
}

template <typename T, size_t dim_>
momentum<T, dim_>::momentum(const T* rest, const size_t dof, const Matrix<T, -1, 1>& mass_vec, const T& dt)
    : dof_(dof), vk_(Matrix<T, -1, 1>::Zero(dim_ * dof)), dt_(dt), d1dt_(1 / dt), d1dtdt_(1 / dt / dt), mass_vec_(dim_ * dof)
{

    dispk_ = Eigen::Map<const Matrix<T, -1, 1>>(rest, dim_ * dof);
#pragma omp parallel for
    for (size_t i = 0; i < dof; ++i)
    {
        for (size_t j = 0; j < dim_; ++j)
        {
            mass_vec_(i * dim_ + j) = mass_vec(i);
        }
    }
}

template <typename T, size_t dim_>
int momentum<T, dim_>::Val(const T* x, data_ptr<T, dim_>& data) const
{
    Eigen::Map<const Matrix<T, -1, 1>> _x(x, dim_ * dof_);
    const Matrix<T, -1, 1>             acce = (_x - dispk_) * d1dt_ - vk_;
    data->save_val(0.5 * acce.dot(mass_vec_.cwiseProduct(acce)));

    return 0;
}
template <typename T, size_t dim_>
int momentum<T, dim_>::Gra(const T* x, data_ptr<T, dim_>& data) const
{
    Eigen::Map<const Matrix<T, -1, 1>> _x(x, dim_ * dof_);

    const Matrix<T, -1, 1> acce = (_x - dispk_) * d1dtdt_ - vk_ * d1dt_;
    data->save_gra(mass_vec_.cwiseProduct(acce));

    return 0;
}

template <typename T, size_t dim_>
int momentum<T, dim_>::Hes(const T* x, data_ptr<T, dim_>& data) const
{
#pragma omp parallel for
    for (size_t i = 0; i < dim_ * dof_; ++i)
        data->save_hes(i, i, d1dtdt_ * mass_vec_(i));

    return 0;
}

template <typename T, size_t dim_>
int momentum<T, dim_>::update_location_and_velocity(const T* new_dispk_ptr, const T* new_velo_ptr)
{
    Eigen::Map<const Matrix<T, -1, 1>> new_dispk(new_dispk_ptr, dim_ * dof_);
    if (new_velo_ptr == nullptr)
        vk_ = (new_dispk - dispk_) * d1dt_;
    else
        vk_ = Map<const Matrix<T, -1, 1>>(new_velo_ptr, dim_ * dof_);

    dispk_ = new_dispk;
    return 0;
}

template <typename T, size_t dim_>
int momentum<T, dim_>::set_initial_velocity(const Matrix<T, dim_, 1>& velo)
{
    Eigen::Map<Matrix<T, -1, -1>> myvelo(vk_.data(), dim_, dof_);
    for (size_t i = 0; i < dim_; ++i)
    {
        myvelo.row(i) = Matrix<T, 1, -1>::Ones(dof_) * velo(i);
    }
    return 0;
}

/******************************************momentum*******************************/
/******************************************position_constraint*******************************/
template <typename T, size_t dim_>
position_constraint<T, dim_>::position_constraint(const size_t dof, const T& w, const vector<size_t>& cons)
    : w_(w), cons_(cons), dof_(dof)
{
    rest_ = Matrix<T, -1, -1>::Zero(dim_, dof);
}

template <typename T, size_t dim_>
position_constraint<T, dim_>::position_constraint(const T* rest, const size_t dof, const T& w, const std::vector<size_t>& cons)
    : w_(w), cons_(cons), dof_(dof)
{
    rest_ = Eigen::Map<const Matrix<T, -1, -1>>(rest, dim_, dof_);
}

//TODO: simplify _x
template <typename T, size_t dim_>
int position_constraint<T, dim_>::Val(const T* x, data_ptr<T, dim_>& data) const
{
    Eigen::Map<const Matrix<T, -1, -1>> deformed(x, dim_, dof_);
    Matrix<T, -1, -1>                   _x = deformed - rest_;
    for (auto iter_c = cons_.begin(); iter_c != cons_.end(); ++iter_c)
    {
        data->save_val(w_ * _x.col(*iter_c).dot(_x.col(*iter_c)));
    }

    return 0;
}

template <typename T, size_t dim_>
int position_constraint<T, dim_>::Gra(const T* x, data_ptr<T, dim_>& data) const
{
    Eigen::Map<const Matrix<T, -1, -1>> deformed(x, dim_, dof_);
    Matrix<T, -1, -1>                   _x = deformed - rest_;

    for (auto iter_c = cons_.begin(); iter_c != cons_.end(); ++iter_c)
        data->save_gra(*iter_c, 2.0 * w_ * _x.col(*iter_c));
    return 0;
}

template <typename T, size_t dim_>
int position_constraint<T, dim_>::Hes(const T* x, data_ptr<T, dim_>& data) const
{
    for (auto iter_c = cons_.begin(); iter_c != cons_.end(); ++iter_c)
    {
        for (size_t j = 0; j < dim_; ++j)
        {
            data->save_hes(*iter_c * dim_ + j, *iter_c * dim_ + j, 2 * w_);
        }
    }
    return 0;
}

template <typename T, size_t dim_>
size_t position_constraint<T, dim_>::Nx() const
{
    return dim_ * dof_;
}
/******************************************position_constraint*******************************/

/******************************************gravity*******************************/
//dof here is not about dim_
template <typename T, size_t dim_>
gravity_energy<T, dim_>::gravity_energy(const size_t dof, const T& w_g, const T& gravity, const Matrix<T, -1, 1>& mass, const char& axis)
    : w_g_(w_g), dof_(dof), gravity_(gravity), mass_(mass), axis_(axis)
{
}

template <typename T, size_t dim_>
int gravity_energy<T, dim_>::Val(const T* x, data_ptr<T, dim_>& data) const
{

    Eigen::Map<const Matrix<T, -1, -1>> _x(x, dim_, dof_);
    size_t                              which_axis = size_t(axis_ - 'x');
    data->save_val((_x.row(which_axis).transpose().array() * mass_.array()).sum() * w_g_ * gravity_);
    return 0;
}

template <typename T, size_t dim_>
int gravity_energy<T, dim_>::Gra(const T* x, data_ptr<T, dim_>& data) const
{
    size_t which_axis = size_t(axis_ - 'x');

    Matrix<T, -1, -1> g(dim_, dof_);
    g.setZero();
    g.row(which_axis) = Matrix<T, -1, 1>::Constant(dof_, gravity_ * w_g_).cwiseProduct(mass_).transpose();
    Eigen::Map<const Matrix<T, -1, 1>> g_(g.data(), dim_ * dof_);
    data->save_gra(g_);
    return 0;
}
template <typename T, size_t dim_>
int gravity_energy<T, dim_>::Hes(const T* x, data_ptr<T, dim_>& data) const
{
    return 0;
}

template <typename T, size_t dim_>
size_t gravity_energy<T, dim_>::Nx() const
{
    return dim_ * dof_;
}
/******************************************gravity*******************************/

/*************************************collision*********************************/
template <typename T, size_t dim_>
collision<T, dim_>::collision(const size_t dof_, const T& w_coll, const char& ground_axis, const T& ground_pos, const size_t& num_surf_point, const shared_ptr<Matrix<T, -1, -1>>& init_points_ptr)
    : ground_axis_(ground_axis), ground_pos_(ground_pos), w_coll_(w_coll), num_surf_point_(num_surf_point), dof_(dof_), init_points_ptr_(init_points_ptr)
{
}

template <typename T, size_t dim_>
int collision<T, dim_>::Val(const T* x, data_ptr<T, dim_>& data) const
{
    const size_t which_axis = size_t(ground_axis_ - 'x');

    Eigen::Map<const Matrix<T, -1, -1>> _x(x, dim_, dof_);
#pragma omp parallel for
    for (size_t i = 0; i < dof_; ++i)
    {
        const T position_now = _x(which_axis, i) + (*init_points_ptr_)(which_axis, i);
        if ((position_now - ground_pos_) < 0)
        {

            data->save_val(w_coll_ * pow((ground_pos_ - position_now), 2));
        }
    }
    return 0;
}
template <typename T, size_t dim_>
int collision<T, dim_>::Gra(const T* x, data_ptr<T, dim_>& data) const
{
    const size_t which_axis = size_t(ground_axis_ - 'x');

    Eigen::Map<const Matrix<T, -1, -1>> _x(x, dim_, dof_);
#pragma omp parallel for
    for (size_t i = 0; i < dof_; ++i)
    {
        const T position_now = _x(which_axis, i) + (*init_points_ptr_)(which_axis, i);
        if ((position_now - ground_pos_) < 0)
        {
            data->save_gra(i * dim_ + which_axis, 2 * w_coll_ * (position_now - ground_pos_));
        }
    }
    return 0;
}
template <typename T, size_t dim_>
int collision<T, dim_>::Hes(const T* x, data_ptr<T, dim_>& data) const
{
    const size_t                        which_axis = size_t(ground_axis_ - 'x');
    Eigen::Map<const Matrix<T, -1, -1>> _x(x, dim_, dof_);
#pragma omp parallel for
    for (size_t i = 0; i < dof_; ++i)
    {
        const T position_now = _x(which_axis, i) + (*init_points_ptr_)(which_axis, i);
        if ((position_now - ground_pos_) < 0)
        {
            for (size_t j = 0; j < dim_; ++j)
            {
                data->save_hes(i * dim_ + j, i * dim_ + j, 2 * w_coll_);
            }
        }
    }

    return 0;
}
template <typename T, size_t dim_>
size_t collision<T, dim_>::Nx() const
{
    return dim_ * dof_;
}

/*************************************collision*********************************/

template class position_constraint<double, 3>;
template class position_constraint<float, 3>;
template class position_constraint<float, 1>;
template class position_constraint<double, 1>;
template class momentum<double, 3>;
template class momentum<float, 3>;
template class gravity_energy<double, 3>;
template class gravity_energy<float, 3>;
// template class collision<FLOAT_TYPE, 3>;

}  //namespace PhysIKA
