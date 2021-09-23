/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity energy for finite element method.
 * @version    : 1.0
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <set>
#include <fstream>

#include "Common/FEMCommonDef.h"
#include "Common/FEMCommonDataStream.h"
#include "Common/FEMCommonEigenExt.h"
#include "Common/FEMCommonType.h"

#include "FEMModelFem.h"

namespace PhysIKA {
/**
 * @brief Compute the lame coeffs
 * 
 * @tparam T 
 * @param Ym 
 * @param Pr 
 * @param mu 
 * @param lambda 
 */
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
    virtual ~elas_intf() {}
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
    using base_class = finite_element<T, dim_, dim_, num_per_cell_, bas_order_, qdrt_axis_, CSTTT, BASIS, QDRT>;
    /**
     * @brief Construct a new Base Elas object
     * 
     * @param nods 
     * @param cells 
     * @param ym 
     * @param poi 
     */
    BaseElas(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const T& ym, const T& poi);

    /**
     * @brief Construct a new Base Elas object
     * 
     * @param nods 
     * @param cells 
     * @param ym 
     * @param poi 
     */
    BaseElas(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const VEC<T>& ym, const VEC<T>& poi);
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

/**
 * @brief Generate the elas energy
 * 
 * @tparam T 
 * @param type 
 * @param csttt_type 
 * @param nods 
 * @param cells 
 * @param Young 
 * @param poi 
 * @param energy 
 * @param intf 
 * @return int 
 */
template <typename T>
int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<T, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const T& Young, const T& poi, std::shared_ptr<Functional<T, 3>>& energy, std::shared_ptr<elas_intf<T, 3>>* intf);

/**
 * @brief Generate the elas energy
 * 
 * @tparam T 
 * @param type 
 * @param csttt_type 
 * @param nods 
 * @param cells 
 * @param Young 
 * @param poi 
 * @param energy 
 * @param intf 
 * @return int 
 */
template <typename T>
int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<T, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const Eigen::Ref<VEC<T>>& Young, const Eigen::Ref<VEC<T>>& poi, std::shared_ptr<Functional<T, 3>>& energy, std::shared_ptr<elas_intf<T, 3>>* intf);

}  // namespace PhysIKA
