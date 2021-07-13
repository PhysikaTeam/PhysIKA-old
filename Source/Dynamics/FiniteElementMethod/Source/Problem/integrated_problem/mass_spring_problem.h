/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: simple mass spring problem
 * @version    : 1.0
 */
#ifndef PhysIKA_GEN_MS_PROBLEM
#define PhysIKA_GEN_MS_PROBLEM
#include <boost/property_tree/ptree.hpp>

#include "Common/framework.h"
#include "Problem/constraint/constraints.h"
#include "Problem/energy/basic_energy.h"
#include "semi_wrapper.h"

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
    ms_problem_builder(const T* x, const boost::property_tree::ptree& pt);
    ms_problem_builder() {}
    std::shared_ptr<Problem<T, 3>> build_problem() const;

    virtual int update_problem(const T* x, const T* v = nullptr);

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
#endif
