/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: main body for finite element method.
 * @version    : 1.0
 */
#ifndef PhysIKA_FEM
#define PhysIKA_FEM

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include "Common/def.h"
#include "Common/eigen_ext.h"

#include "constitutive.h"
#include "gaussian_quadrature.h"
#include "basis_func.h"

namespace PhysIKA {
template <typename T, int row, int col>
using vec_vec_mat = std::vector<std::vector<Eigen::Matrix<T, row, col>, Eigen::aligned_allocator<Eigen::Matrix<T, row, col>>>>;

/**
 * finite element class definition.
 *
 * Sample usage:
 * finite_element energy = new finite_element(nods, cells);
 * energy->Val(x, data);
 * energy->Gra(x, data);
 * energy->Hes(x, data);
 */
template <typename T, size_t dim_, size_t field_, size_t num_per_cell_, size_t bas_order_, size_t qdrt_axis_, template <typename, size_t, size_t> class CSTTT,  // constituitive function
          template <typename, size_t, size_t, size_t, size_t>
          class BASIS,  //  basis
          template <typename, size_t, size_t, size_t>
          class QDRT>  //
class finite_element : public Functional<T, field_>
{
public:
    using basis = BASIS<T, dim_, field_, bas_order_, num_per_cell_>;
    using csttt = CSTTT<T, dim_, field_>;
    using qdrt  = QDRT<T, dim_, qdrt_axis_, num_per_cell_>;
    finite_element(const Eigen::Matrix<T, dim_, -1>&            nods,
                   const Eigen::Matrix<int, num_per_cell_, -1>& cells);
    size_t                   Nx() const;
    int                      Val(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const;
    int                      Gra(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const;
    int                      Hes(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const;
    Eigen::Matrix<T, -1, -1> mtr_;

protected:  //about elements
    const size_t                                all_dim_, num_nods_, num_cells_, num_qdrt_;
    const Eigen::Matrix<T, dim_, -1>            nods_;   // vertices
    const Eigen::Matrix<int, num_per_cell_, -1> cells_;  // elements
    const Eigen::Matrix<int, dim_, 1>           dim_all_rows_;
    const Eigen::Matrix<int, field_, 1>         field_all_rows_;
    const qdrt                                  quadrature_;

protected:  // precomputed values
    void                                                  PreComputation();
    vec_vec_mat<T, dim_, dim_>                            Dm_inv_;
    std::vector<std::vector<T>>                           Jac_det_;
    vec_vec_mat<T, field_ * dim_, field_ * num_per_cell_> Ddef_Dx_;
    vec_vec_mat<T, num_per_cell_, dim_>                   Dphi_Dxi_;
};

#define FEM_TEMP template <typename T, size_t dim_, size_t field_, size_t num_per_cell_, size_t bas_order_, size_t qdrt_axis_, template <typename, size_t, size_t> class CSTTT, template <typename, size_t, size_t, size_t, size_t> class BASIS, template <typename, size_t, size_t, size_t> class QDRT>
#define FEM_CLASS finite_element<T, dim_, field_, num_per_cell_, bas_order_, qdrt_axis_, CSTTT, BASIS, QDRT>

}  // namespace PhysIKA
#endif
