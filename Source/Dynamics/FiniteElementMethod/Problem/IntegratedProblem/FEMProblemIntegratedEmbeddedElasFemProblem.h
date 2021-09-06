/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: embedded elasticity finite element method problem
 * @version    : 1.0
 */
#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Common/FEMCommonFramework.h"
#include "Problem/Constraint/FEMProblemConstraints.h"
#include "Problem/Energy/FEMProblemEnergyBasicEnergy.h"
#include "Model/FEM/FEMModelFemElasEnergy.h"
#include "Geometry/FEMGeometryEmbeddedInterpolate.h"
#include "FEMProblemIntegratedProblemSemiWrapper.h"

namespace PhysIKA {
/**
 * embedded elasticity problem builder, build the embeded elasticity problem
 *
 */
template <typename T>
class embedded_elas_problem_builder : public embedded_problem_builder<T, 3>
    , public semi_wrapper<T>
{
public:
    /**
     * @brief Construct a new embedded_elas_problem_builder object
     * 
     * @param x 
     * @param pt 
     */
    embedded_elas_problem_builder(const T* x, const boost::property_tree::ptree& pt);

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

    /**
     * @brief Get the coarse to fine coefficient object
     * 
     * @return const Eigen::SparseMatrix<T>& 
     */
    const Eigen::SparseMatrix<T>& get_coarse_to_fine_coefficient() const
    {
        return coarse_to_fine_coef_;
    }

    /**
     * @brief Get the fine to coarse coefficient object
     * 
     * @return const Eigen::SparseMatrix<T>& 
     */
    const Eigen::SparseMatrix<T>& get_fine_to_coarse_coefficient() const
    {
        return fine_to_coarse_coef_;
    }

    /**
     * @brief Get the elas energy object
     * 
     * @return std::shared_ptr<elas_intf<T, 3>> 
     */
    std::shared_ptr<elas_intf<T, 3>> get_elas_energy() const
    {
        return elas_intf_;
    }

    /**
     * @brief Get the embedded interpolate object
     * 
     * @return std::shared_ptr<embedded_interpolate<T>> 
     */
    std::shared_ptr<embedded_interpolate<T>> get_embedded_interpolate()
    {
        return embedded_interp_;
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

private:
    Eigen::Matrix<T, -1, -1>                 REST_;
    Eigen::MatrixXi                          cells_;
    int                                      fine_verts_num_;
    std::shared_ptr<embedded_interpolate<T>> embedded_interp_;

    Eigen::SparseMatrix<T> coarse_to_fine_coef_;  // coarse * coef = fine
    Eigen::SparseMatrix<T> fine_to_coarse_coef_;  // fine * coef = coarse

    std::shared_ptr<constraint_4_coll<T>>          collider_;
    std::shared_ptr<momentum<T, 3>>                kinetic_;
    std::vector<std::shared_ptr<Functional<T, 3>>> ebf_;
    std::vector<std::shared_ptr<Constraint<T>>>    cbf_;
    std::shared_ptr<elas_intf<T, 3>>               elas_intf_;

    const boost::property_tree::ptree pt_;
};

}  // namespace PhysIKA
