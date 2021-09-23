/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: some basic energy function.
 * @version    : 1.0
 */
#pragma once

#include <vector>
#include <memory>
#include "Common/FEMCommonDef.h"
#include "Common/FEMCommonDataStream.h"

namespace PhysIKA {

template <typename T, size_t dim_>
using data_ptr = std::shared_ptr<dat_str_core<T, dim_>>;

/**
 * Position constraint class, turn the position constraint to soft energy term.
 *
 */
template <typename T, size_t dim_>
class position_constraint : public Functional<T, dim_>
{
public:
    // used for displace based
    /**
     * @brief Construct a new position constraint object
     * 
     * @param dof 
     * @param w 
     * @param cons 
     */
    position_constraint(const size_t dof, const T& w, const std::vector<size_t>& cons);

    // used for position based
    /**
     * @brief Construct a new position constraint object
     * 
     * @param rest 
     * @param dof 
     * @param w 
     * @param cons 
     */
    position_constraint(const T* rest, const size_t dof, const T& w, const std::vector<size_t>& cons);

    /**
     * @brief Get the value
     * 
     * @param x 
     * @param data 
     * @return int 
     */
    int Val(const T* x, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the gradients
     * 
     * @param x 
     * @param data 
     * @return int 
     */
    int Gra(const T* x, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the hessian
     * 
     * @param x 
     * @param data 
     * @return int 
     */
    int Hes(const T* x, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the number of dimension
     * 
     * @return size_t 
     */
    size_t Nx() const;

private:
    Eigen::Matrix<T, -1, -1>  rest_;
    const size_t              dof_;
    const T                   w_;
    const std::vector<size_t> cons_;
};

/**
 * gravity energy term, Gravitational potential energy
 *
 */
template <typename T, size_t dim_>
class gravity_energy : public Functional<T, dim_>
{
public:
    /**
     * @brief Construct a new gravity energy object
     * 
     * @param dof 
     * @param w_g 
     * @param gravity 
     * @param mass 
     * @param axis 
     */
    gravity_energy(const size_t dof, const T& w_g, const T& gravity, const Eigen::Matrix<T, -1, 1>& mass, const char& axis);

    /**
     * @brief Get the value
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Val(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the gradients
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Gra(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the hesssian
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Hes(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the number of dimension
     * 
     * @return size_t 
     */
    size_t Nx() const;

private:
    const char                    axis_;
    const size_t                  dof_;
    const T                       w_g_;
    const T                       gravity_;
    const Eigen::Matrix<T, -1, 1> mass_;
};

/**
 * turn collision term to soft energy term.
 *
 */
template <typename T, size_t dim_>
class collision : public Functional<T, dim_>
{
public:
    /**
     * @brief Construct a new collision object
     * 
     * @param dim 
     * @param w_coll 
     * @param ground_axis 
     * @param ground_pos 
     * @param num_surf_point 
     * @param init_points_ptr 
     */
    collision(const size_t dim, const T& w_coll, const char& ground_axis, const T& ground_pos, const size_t& num_surf_point, const std::shared_ptr<Eigen::Matrix<T, -1, -1>>& init_points_ptr);

    /**
     * @brief Get the value
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Val(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the gradients
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Gra(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the hessian
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Hes(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the Number of dimension
     * 
     * @return size_t 
     */
    size_t Nx() const;

private:
    const char                                      ground_axis_;
    const T                                         w_coll_;
    const T                                         ground_pos_;
    const size_t                                    num_surf_point_;
    const size_t                                    dof_;
    const std::shared_ptr<Eigen::Matrix<T, -1, -1>> init_points_ptr_;
};

/**
 * momentum energy term.
 *
 */
template <typename T, size_t dim_>
class momentum : public Functional<T, dim_>
{
public:
    // used for displacement based
    /**
     * @brief Construct a new momentum object
     * 
     * @param dof 
     * @param mass_vec 
     * @param dt 
     */
    momentum(const size_t dof, const Eigen::Matrix<T, -1, 1>& mass_vec, const T& dt);

    // used for position based
    /**
     * @brief Construct a new momentum object
     * 
     * @param x 
     * @param dof 
     * @param mass_vec 
     * @param dt 
     */
    momentum(const T* x, const size_t dof, const Eigen::Matrix<T, -1, 1>& mass_vec, const T& dt);

    /**
     * @brief Get the value
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Val(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the gradients
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Gra(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Get the hession
     * 
     * @param disp 
     * @param data 
     * @return int 
     */
    int Hes(const T* disp, data_ptr<T, dim_>& data) const;

    /**
     * @brief Update the location and velocity of the object
     * 
     * @param new_dispk_ptr 
     * @param new_velo_ptr 
     * @return int 
     */
    int update_location_and_velocity(const T* new_dispk_ptr, const T* new_velo_ptr);

    /**
     * @brief Set the initial velocity of the object
     * 
     * @param velo 
     * @return int 
     */
    int set_initial_velocity(const Eigen::Matrix<T, dim_, 1>& velo);

    /**
     * @brief Update the object
     * 
     * @param x 
     * @return int 
     */
    int update(const T* x);

    /**
     * @brief Get the number of dimension
     * 
     * @return size_t 
     */
    size_t Nx() const;

    /**
     * @brief Get the mass vec object
     * 
     * @return Eigen::Matrix<T, -1, 1> 
     */
    Eigen::Matrix<T, -1, 1> get_mass_vec() const
    {
        return mass_vec_;
    }

    Eigen::Matrix<T, -1, 1> vk_, dispk_;

private:
    const size_t            dof_;
    Eigen::Matrix<T, -1, 1> mass_vec_;
    const T                 dt_;
    const T                 d1dt_;
    const T                 d1dtdt_;
};

}  //namespace PhysIKA
