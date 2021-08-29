/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: posisson equation for finite element method.
 * @version    : 1.0
 */
#ifndef PhysIKA_POISSON_FEM
#define PhysIKA_POISSON_FEM

#include "Common/def.h"

#include "gaussian_quadrature.h"
#include "basis_func.h"
#include "FEM.h"

namespace PhysIKA {

/**
 * poisson equation class
 *
 */
template <typename T, size_t dim_, size_t num_per_cell_, size_t qdrt_axis_, template <typename, size_t, size_t> class CSTTT,  // constituitive function
          template <typename, size_t, size_t, size_t, size_t>
          class BASIS,  //  basis
          template <typename, size_t, size_t, size_t>
          class QDRT>  //
class POISSON : public finite_element<T, dim_, 1, num_per_cell_, 1, qdrt_axis_, CSTTT, BASIS, QDRT>
{
public:
    using base_class = finite_element<T, dim_, 1, num_per_cell_, 1, qdrt_axis_, CSTTT, BASIS, QDRT>;
    POISSON(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const Eigen::Matrix<T, 1, -1>& mtr)
        : base_class(nods, cells)
    {
        base_class::mtr_ = mtr;
    }
};
using HEX_POISSON  = POISSON<double, 3, 8, 2, quadratic_csttt, basis_func, quadrature>;
using QUAD_POISSON = POISSON<double, 2, 4, 2, quadratic_csttt, basis_func, quadrature>;

}  // namespace PhysIKA
#endif
