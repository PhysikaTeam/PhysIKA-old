/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: semi imicilpt euler solver.
 * @version    : 1.0
 */
#pragma once

#include "Solver/FEMSolverSemiImplicitEuler.h"

/**
 * semi implicit method wrapper.
 *
 */
template <typename T>
class semi_wrapper
{
public:
    /**
     * @brief Construct a new semi_wrapper object
     * 
     */
    semi_wrapper()
        : semi_implicit_(nullptr) {}

    /**
     * @brief Get the semi implicit object
     * 
     * @return std::shared_ptr<semi_implicit<T>> 
     */
    virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const
    {
        return semi_implicit_;
    }

protected:
    std::shared_ptr<semi_implicit<T>> semi_implicit_;
};
