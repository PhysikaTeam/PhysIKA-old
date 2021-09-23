/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: fast mass spring problem
 * @version    : 1.0
 */
#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Common/FEMCommonFramework.h"
#include "Problem/Constraint/FEMProblemConstraints.h"
#include "Problem/Energy/FEMProblemEnergyBasicEnergy.h"
#include "Geometry/FEMGeometryEmbeddedInterpolate.h"
#include "FEMProblemIntegratedMassSpringProblem.h"
#include "FEMProblemIntegratedEmbeddedMassSpringProblem.h"
#include "Solver/FEMSolverFastMsSolver.h"

namespace PhysIKA {
/**
 * fast mass spring problem builder, build the fast mass spring problem
 *
 */
template <typename T>
class fast_ms_builder : public embedded_ms_problem_builder<T>
{
public:
    /**
     * @brief Construct a new fast_ms_builder object
     * 
     * @param x 
     * @param pt 
     */
    fast_ms_builder(const T* x, const boost::property_tree::ptree& pt);

    /**
     * @brief Get the fast ms solver info object
     * 
     * @return std::shared_ptr<fast_ms_info<T>> 
     */
    std::shared_ptr<fast_ms_info<T>> get_fast_ms_solver_info() const
    {
        return solver_info_;
    }

    /**
     * @brief Update the problem object
     * 
     * @param x 
     * @param v 
     * @return int 
     */
    virtual int update_problem(const T* x, const T* v = nullptr);

    using embedded_ms_problem_builder<T>::REST_;
    using embedded_ms_problem_builder<T>::cells_;
    using embedded_ms_problem_builder<T>::collider_;
    using embedded_ms_problem_builder<T>::kinetic_;
    using embedded_ms_problem_builder<T>::ebf_;
    using embedded_ms_problem_builder<T>::cbf_;
    using embedded_ms_problem_builder<T>::pt_;
    using embedded_ms_problem_builder<T>::get_collider;
    using embedded_ms_problem_builder<T>::get_cells;
    using embedded_ms_problem_builder<T>::get_nods;
    using embedded_ms_problem_builder<T>::build_problem;
    using embedded_ms_problem_builder<T>::semi_implicit_;

    using embedded_ms_problem_builder<T>::fine_verts_num_;
    using embedded_ms_problem_builder<T>::embedded_interp_;
    using embedded_ms_problem_builder<T>::coarse_to_fine_coef_;  // coarse * coef = fine
    using embedded_ms_problem_builder<T>::fine_to_coarse_coef_;  // fine * coef = coarse

protected:
    std::shared_ptr<fast_ms_info<T>> solver_info_;
};
}  // namespace PhysIKA
