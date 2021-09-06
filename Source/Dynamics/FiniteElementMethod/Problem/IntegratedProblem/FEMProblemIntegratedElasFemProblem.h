/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity finite element method problem
 * @version    : 1.0
 */
#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Common/FEMCommonFramework.h"
#include "Problem/Constraint/FEMProblemConstraints.h"
#include "Problem/Energy/FEMProblemEnergyBasicEnergy.h"
#include "Model/FEM/FEMModelFemElasEnergy.h"
#include "FEMProblemIntegratedProblemSemiWrapper.h"

namespace PhysIKA {
/**
 * elasticity problem builder, build the elasticity problem.
 *
 */
template <typename T>
class elas_problem_builder : public problem_builder<T, 3>
    , public semi_wrapper<T>
{
public:
    /**
     * @brief Construct a new elas_problem_builder object
     * 
     * @param x 
     * @param pt 
     */
    elas_problem_builder(const T* x, const boost::property_tree::ptree& pt);

    /**
     * @brief Build the object
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
    int update_problem(const T* x, const T* v = nullptr);

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

    using semi_wrapper<T>::semi_implicit_;

    /**
     * @brief Get the elas energy object
     * 
     * @return std::shared_ptr<elas_intf<T, 3>> 
     */
    std::shared_ptr<elas_intf<T, 3>> get_elas_energy() const
    {
        return elas_intf_;
    }

private:
    Eigen::Matrix<T, -1, -1> REST_;
    Eigen::MatrixXi          cells_;

    std::shared_ptr<constraint_4_coll<T>>          collider_;
    std::shared_ptr<momentum<T, 3>>                kinetic_;
    std::vector<std::shared_ptr<Functional<T, 3>>> ebf_;
    std::vector<std::shared_ptr<Constraint<T>>>    cbf_;
    std::shared_ptr<elas_intf<T, 3>>               elas_intf_;

    const boost::property_tree::ptree pt_;
};

}  // namespace PhysIKA
