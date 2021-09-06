/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: simple mass spring problem
 * @version    : 1.0
 */
#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Common/FEMCommonFramework.h"
#include "Problem/Constraint/FEMProblemConstraints.h"
#include "Problem/Energy/FEMProblemEnergyBasicEnergy.h"
#include "FEMProblemIntegratedProblemSemiWrapper.h"

namespace PhysIKA {
/**
 * mass spring problem builder, build the mass spring problem
 *
 */
template <typename T>
class ms_problem_builder : public embedded_problem_builder<T, 3>
    , public semi_wrapper<T>
{
public:
    /**
     * @brief Construct a new ms_problem_builder object
     * 
     * @param x 
     * @param pt 
     */
    ms_problem_builder(const T* x, const boost::property_tree::ptree& pt);

    /**
     * @brief Construct a new ms_problem_builder object
     * 
     */
    ms_problem_builder() {}

    /**
     * @brief Build the problem object
     * 
     * @return std::shared_ptr<Problem<T, 3>> 
     */
    std::shared_ptr<Problem<T, 3>> build_problem() const;

    /**
     * @brief Update the problem object
     * 
     * @param x 
     * @param v 
     * @return int 
     */
    virtual int update_problem(const T* x, const T* v = nullptr);

    /**
     * @brief Get the nods object
     * 
     * @return Eigen::Matrix<T, -1, -1> 
     */
    Eigen::Matrix<T, -1, -1> get_nods() const
    {
        return REST_;
    }

    /**
     * @brief Get the cells object
     * 
     * @return Eigen::MatrixXi 
     */
    Eigen::MatrixXi get_cells() const
    {
        return cells_;
    }

    /**
     * @brief Get the collider object
     * 
     * @return std::shared_ptr<constraint_4_coll<T>> 
     */
    std::shared_ptr<constraint_4_coll<T>> get_collider() const
    {
        return collider_;
    }

    /**
     * @brief Get the semi implicit object
     * 
     * @return std::shared_ptr<semi_implicit<T>> 
     */
    virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const
    {
        return semi_implicit_;
    }

    using semi_wrapper<T>::semi_implicit_;

protected:
    Eigen::Matrix<T, -1, -1> REST_;
    Eigen::MatrixXi          cells_;

    std::shared_ptr<constraint_4_coll<T>>          collider_;
    std::shared_ptr<momentum<T, 3>>                kinetic_;
    std::vector<std::shared_ptr<Functional<T, 3>>> ebf_;
    std::vector<std::shared_ptr<Constraint<T>>>    cbf_;

    boost::property_tree::ptree pt_;
};

}  // namespace PhysIKA
