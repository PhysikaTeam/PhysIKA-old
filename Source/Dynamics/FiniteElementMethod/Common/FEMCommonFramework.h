/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: framework for problem and solver building.
 * @version    : 1.0
 */
#pragma once

#include "Common/FEMCommonDef.h"
#include "Common/FEMCommonDataStream.h"
#include "Geometry/FEMGeometryEmbeddedInterpolate.h"
#include "Solver/FEMSolverSemiImplicitEuler.h"
#include <memory>
#include <iostream>

namespace PhysIKA {
/**
 * the problem interface. A common problem interface for unifying access.
 *
 */
template <typename T, size_t dim_>
class Problem
{
public:
    virtual ~Problem() {}
    Problem(const std::shared_ptr<Functional<T, dim_>>& energy,
            const std::shared_ptr<Constraint<T>>&       constraint)
        : energy_(energy), constraint_(constraint) {}
    std::shared_ptr<Functional<T, dim_>> energy_;
    std::shared_ptr<Constraint<T>>       constraint_;

    /**
     * @brief dim of freedom.
     * 
     * @return size_t 
     */
    size_t Nx() const
    {
        return energy_->Nx();
    }

private:
};

/**
 * the problem builder interface. A common problem interface for unifying access.
 *
 */
template <typename T, size_t dim_>
class problem_builder
{
public:
    virtual ~problem_builder() {}
    /**
     * @brief build the problem.
     * 
     * @return std::shared_ptr<Problem<T, dim_>> 
     */
    virtual std::shared_ptr<Problem<T, dim_>> build_problem() const = 0;
    /**
     * @brief update the problem.
     * 
     * @param x 
     * @param v 
     * @return int 
     */
    virtual int update_problem(const T* x, const T* v = nullptr);
};

/**
 * embedded problem builder interface.
 *
 */
template <typename T, size_t dim_>
class embedded_problem_builder
{
public:
    virtual ~embedded_problem_builder() {}
    /**
     * @brief build the problem.
     * 
     * @return std::shared_ptr<Problem<T, dim_>> 
     */
    virtual std::shared_ptr<Problem<T, dim_>> build_problem() const = 0;
    /**
     * @brief update problem.
     * 
     * @param x 
     * @param v 
     * @return int 
     */
    virtual int update_problem(const T* x, const T* v = nullptr)
    {
        return 0;
    }
    /**
     * @brief Get the embedded interpolate object
     * 
     * @return std::shared_ptr<embedded_interpolate<T>> 
     */
    virtual std::shared_ptr<embedded_interpolate<T>> get_embedded_interpolate()
    {
        return nullptr;
    }
    /**
     * @brief Get the nods object
     * 
     * @return Eigen::Matrix<T, -1, -1> 
     */
    virtual Eigen::Matrix<T, -1, -1> get_nods() const
    {
        return Eigen::Matrix<T, -1, -1>::Zero(0, 0);
    }
    /**
     * @brief Get the semi implicit object
     * 
     * @return std::shared_ptr<semi_implicit<T>> 
     */
    virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const
    {
        return nullptr;
    }
};

/**
 * solver interface, to unifying the calling interface.
 *
 */
template <typename T, size_t dim_>
class solver
{
public:
    solver(const std::shared_ptr<Problem<T, dim_>>& pb, std::shared_ptr<dat_str_core<T, dim_>> dat_str = nullptr)
        : pb_(pb), dat_str_(dat_str == nullptr ? std::make_shared<dat_str_core<T, dim_>>(pb->energy_->Nx() / dim_) : dat_str) {}
    virtual int                                    solve(T* x_star) const = 0;
    const std::shared_ptr<Problem<T, dim_>>        pb_;
    mutable std::shared_ptr<dat_str_core<T, dim_>> dat_str_;
};

}  // namespace PhysIKA
