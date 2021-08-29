/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity energy for finite element method.
 * @version    : 1.0
 */
#ifndef ELAS_ENERGY
#define ELAS_ENERGY

#include <Eigen/Dense>
#include <iostream>
#include <set>
#include <fstream>

#include "Common/def.h"
#include "Common/data_str_core.h"
#include "Common/eigen_ext.h"
#include "Common/BCSR.h"
#include "Common/DEFINE_TYPE.h"

#include "FEM.h"

namespace PhysIKA {
template <typename T>
inline void compute_lame_coeffs(const T Ym, const T Pr, T& mu, T& lambda)
{
    mu     = Ym / (2 * (1 + Pr));
    lambda = Ym * Pr / ((1 + Pr) * (1 - 2 * Pr));
}

/**
 * elasticity interface
 *
 */
template <typename T, size_t dim_>
class elas_intf
{
public:
    using vec_mat = VEC_MAT<Eigen::Matrix<T, dim_, dim_>>;
    virtual ~elas_intf() {}
    virtual int aver_ele_R(const T* x, vec_mat& vec_R) const = 0;
};

/**
 * base elasticity energy class.
 *
 */
template <typename T, size_t dim_, size_t num_per_cell_, size_t bas_order_, size_t qdrt_axis_, template <typename, size_t, size_t> class CSTTT,  // constituitive function
          template <typename, size_t, size_t, size_t, size_t>
          class BASIS,  //  basis
          template <typename, size_t, size_t, size_t>
          class QDRT>  //
class BaseElas : public elas_intf<T, dim_>
    , public finite_element<T, dim_, dim_, num_per_cell_, bas_order_, qdrt_axis_, CSTTT, BASIS, QDRT>
{
public:
    using vec_mat    = typename elas_intf<T, dim_>::vec_mat;
    using base_class = finite_element<T, dim_, dim_, num_per_cell_, bas_order_, qdrt_axis_, CSTTT, BASIS, QDRT>;
    BaseElas(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const T& ym, const T& poi);
    BaseElas(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const VEC<T>& ym, const VEC<T>& poi);

    int aver_ele_R(const T* x, vec_mat& vec_R) const;
};

#define ELAS_CLASS BaseElas<T, dim_, num_per_cell_, bas_order_, qdrt_axis_, CSTTT, BASIS, QDRT>
#define ELAS_TEMP template <typename T, size_t dim_, size_t num_per_cell_, size_t bas_order_, size_t qdrt_axis_, template <typename, size_t, size_t> class CSTTT, template <typename, size_t, size_t, size_t, size_t> class BASIS, template <typename, size_t, size_t, size_t> class QDRT>  //

template <typename T>
using TET_lin_ELAS = BaseElas<T, 3, 4, 1, 1, linear_csttt, basis_func, quadrature>;
template <typename T>
using HEX_lin_ELAS = BaseElas<T, 3, 8, 1, 2, linear_csttt, basis_func, quadrature>;
template <typename T>
using TET_stvk_ELAS = BaseElas<T, 3, 4, 1, 1, stvk, basis_func, quadrature>;
template <typename T>
using HEX_stvk_ELAS = BaseElas<T, 3, 8, 1, 2, stvk, basis_func, quadrature>;
template <typename T>
using TET_corotated_ELAS = BaseElas<T, 3, 4, 1, 1, corotated_csttt, basis_func, quadrature>;
template <typename T>
using HEX_corotated_ELAS = BaseElas<T, 3, 8, 1, 2, corotated_csttt, basis_func, quadrature>;
template <typename T>
using TET_arap_ELAS = BaseElas<T, 3, 4, 1, 1, arap_csttt, basis_func, quadrature>;
template <typename T>
using HEX_arap_ELAS = BaseElas<T, 3, 8, 1, 2, arap_csttt, basis_func, quadrature>;

template <typename T>
int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<T, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const T& Young, const T& poi, std::shared_ptr<Functional<T, 3>>& energy, std::shared_ptr<elas_intf<T, 3>>* intf);

template <typename T>
int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<T, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const Eigen::Ref<VEC<T>>& Young, const Eigen::Ref<VEC<T>>& poi, std::shared_ptr<Functional<T, 3>>& energy, std::shared_ptr<elas_intf<T, 3>>* intf);

}  // namespace PhysIKA
#endif
