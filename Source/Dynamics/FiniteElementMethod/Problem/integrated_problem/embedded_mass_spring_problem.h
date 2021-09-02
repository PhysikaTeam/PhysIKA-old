/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: embedded elasticity mass spring method problem
 * @version    : 1.0
 */
#ifndef PhysIKA_EMBEDDED_MS_PROBLEM
#define PhysIKA_EMBEDDED_MS_PROBLEM
#include <boost/property_tree/ptree.hpp>

#include "Common/framework.h"
#include "Problem/constraint/constraints.h"
#include "Problem/energy/basic_energy.h"
#include "Geometry/embedded_interpolate.h"
#include "mass_spring_problem.h"

namespace PhysIKA {

/**
 * embedded mass spring problem builder, build the embeded mass spring problem.
 *
 */
template <typename T>
class embedded_ms_problem_builder : public ms_problem_builder<T>
{
public:
    embedded_ms_problem_builder(const T* x, const boost::property_tree::ptree& pt);
    embedded_ms_problem_builder() {}

    virtual int update_problem(const T* x, const T* v = nullptr);

    std::shared_ptr<embedded_interpolate<T>> get_embedded_interpolate()
    {
        return embedded_interp_;
    }
    virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const
    {
        return semi_implicit_;
    }

    using ms_problem_builder<T>::REST_;
    using ms_problem_builder<T>::cells_;
    using ms_problem_builder<T>::collider_;
    using ms_problem_builder<T>::kinetic_;
    using ms_problem_builder<T>::ebf_;
    using ms_problem_builder<T>::cbf_;
    using ms_problem_builder<T>::pt_;
    using ms_problem_builder<T>::get_collider;
    using ms_problem_builder<T>::get_cells;
    using ms_problem_builder<T>::get_nods;
    using ms_problem_builder<T>::build_problem;
    using ms_problem_builder<T>::semi_implicit_;

protected:
    int                                      fine_verts_num_;
    std::shared_ptr<embedded_interpolate<T>> embedded_interp_;

    Eigen::SparseMatrix<T> coarse_to_fine_coef_;  // coarse * coef = fine
    Eigen::SparseMatrix<T> fine_to_coarse_coef_;  // fine * coef = coarse
};
}  // namespace PhysIKA
#endif
