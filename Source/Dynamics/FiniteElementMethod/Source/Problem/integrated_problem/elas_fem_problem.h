/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity finite element method problem
 * @version    : 1.0
 */
#ifndef PhysIKA_GEN_ELAS_PROBLEM
#define PhysIKA_GEN_ELAS_PROBLEM
#include <boost/property_tree/ptree.hpp>

#include "Common/framework.h"
#include "Problem/constraint/constraints.h"
#include "Problem/energy/basic_energy.h"
#include "Model/fem/elas_energy.h"
#include "semi_wrapper.h"

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
    elas_problem_builder(const T* x, const boost::property_tree::ptree& pt);
    std::shared_ptr<Problem<T, 3>> build_problem() const;

    int update_problem(const T* x, const T* v = nullptr);

    Eigen::Matrix<T, -1, -1> get_nods() const
    {
        return REST_;
    }
    Eigen::MatrixXi get_cells() const
    {
        return cells_;
    }
    std::shared_ptr<constraint_4_coll<T>> get_collider() const
    {
        return collider_;
    }

    using semi_wrapper<T>::semi_implicit_;

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
#endif
