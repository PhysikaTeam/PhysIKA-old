/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: main body for finite element method.
 * @version    : 1.0
 */
#include <iostream>

#include "Common/eigen_ext.h"

#include "FEM.h"
using namespace std;
using namespace Eigen;
namespace PhysIKA {

FEM_TEMP
FEM_CLASS::finite_element(const Matrix<T, dim_, -1>& nods, const Matrix<int, num_per_cell_, -1>& cells)
    : all_dim_(nods.cols() * field_), num_nods_(nods.cols()), num_cells_(cells.cols()), nods_(nods), cells_(cells), dim_all_rows_(Matrix<int, dim_, 1>::LinSpaced(dim_, 0, dim_ - 1)), field_all_rows_(Matrix<int, field_, 1>::LinSpaced(field_, 0, field_ - 1)), quadrature_(), num_qdrt_(static_cast<size_t>(pow(qdrt_axis_, dim_)))
{
    static_assert(std::is_base_of<constitutive<T, dim_, field_>, csttt>::value, "CSTTT must derive from elas_csttt");
    static_assert(std::is_base_of<basis_func<T, dim_, field_, bas_order_, num_per_cell_>, basis>::value, "BASIS must derive from basis_func");
    static_assert(std::is_base_of<quadrature<T, dim_, qdrt_axis_, num_per_cell_>, qdrt>::value, "GAUS must derive from gaus_quad");
    PreComputation();
}

FEM_TEMP
size_t FEM_CLASS::Nx() const
{
    return all_dim_;
}

FEM_TEMP
void FEM_CLASS::PreComputation()
{
    Dm_inv_.resize(num_cells_);
    Jac_det_.resize(num_cells_);
    Ddef_Dx_.resize(num_cells_);
    Dphi_Dxi_.resize(num_cells_);

    Eigen::Matrix<T, dim_, dim_>                            Dm_inv_tmp;
    T                                                       Jac_det_tmp;
    Eigen::Matrix<T, field_ * dim_, field_ * num_per_cell_> Ddef_Dx_tmp;
    Eigen::Matrix<T, num_per_cell_, dim_>                   Dphi_Dxi_tmp;

    for (size_t cell_id = 0; cell_id < num_cells_; ++cell_id)
    {
        const Matrix<T, dim_, num_per_cell_> X_cell = indexing(nods_, dim_all_rows_, cells_.col(cell_id));
        for (size_t qdrt_id = 0; qdrt_id < num_qdrt_; ++qdrt_id)
        {
            basis::calc_Dphi_Dxi(quadrature_.PNT_.col(qdrt_id), X_cell.data(), Dphi_Dxi_tmp);
            Dphi_Dxi_[cell_id].push_back(Dphi_Dxi_tmp);

            basis::calc_InvDm_Det(Dphi_Dxi_tmp, X_cell.data(), Jac_det_tmp, Dm_inv_tmp);
            Jac_det_[cell_id].push_back(Jac_det_tmp);
            Dm_inv_[cell_id].push_back(Dm_inv_tmp);
            basis::get_Ddef_Dx(Dphi_Dxi_tmp, Dm_inv_tmp, Ddef_Dx_tmp);
            Ddef_Dx_[cell_id].push_back(Ddef_Dx_tmp);
        }
    }
    return;
}

FEM_TEMP
int FEM_CLASS::Val(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const
{
    Eigen::Map<const Eigen::Matrix<T, -1, -1>> deformed(x, field_, num_nods_);
#pragma omp parallel for
    for (size_t cell_id = 0; cell_id < num_cells_; ++cell_id)
    {
        Matrix<T, field_, dim_>                def_gra;
        const Matrix<T, field_, num_per_cell_> x_cell = indexing(deformed, field_all_rows_, cells_.col(cell_id));
        for (size_t qdrt_id = 0; qdrt_id < num_qdrt_; ++qdrt_id)
        {
            basis::get_def_gra(Dphi_Dxi_[cell_id][qdrt_id], x_cell.data(), Dm_inv_[cell_id][qdrt_id], def_gra);
            data->save_val(csttt::val(def_gra, mtr_.col(cell_id)) * quadrature_.WGT_[qdrt_id] * Jac_det_[cell_id][qdrt_id]);
        }
    }

    return 0;
}

FEM_TEMP
int FEM_CLASS::Gra(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const
{
    Eigen::Map<const Eigen::Matrix<T, -1, -1>> deformed(x, field_, num_nods_);
#pragma omp parallel for
    for (size_t cell_id = 0; cell_id < num_cells_; ++cell_id)
    {
        Matrix<T, field_, dim_> def_gra;

        const Matrix<T, field_, num_per_cell_> x_cell = indexing(deformed, field_all_rows_, cells_.col(cell_id));

        Matrix<T, field_ * dim_, 1>          gra_F_based;
        Matrix<T, field_ * num_per_cell_, 1> gra_x_based = Matrix<T, field_ * num_per_cell_, 1>::Zero();

        //TODO:considering the order of basis
        for (size_t qdrt_id = 0; qdrt_id < num_qdrt_; ++qdrt_id)
        {
            basis::get_def_gra(Dphi_Dxi_[cell_id][qdrt_id], x_cell.data(), Dm_inv_[cell_id][qdrt_id], def_gra);
            gra_F_based = csttt::gra(def_gra, mtr_.col(cell_id));
            gra_x_based.noalias() += Ddef_Dx_[cell_id][qdrt_id].transpose() * gra_F_based * quadrature_.WGT_[qdrt_id] * Jac_det_[cell_id][qdrt_id];
        }

        //save gra
        const Eigen::Map<Matrix<T, field_, num_per_cell_>> gra_x_based_reshape(gra_x_based.data());
        for (size_t p = 0; p < num_per_cell_; ++p)
        {
            data->save_gra(cells_(p, cell_id), gra_x_based_reshape.col(p));
        }
    }

    return 0;
}

FEM_TEMP
int FEM_CLASS::Hes(const T* x, std::shared_ptr<dat_str_core<T, field_>>& data) const
{
    Eigen::Map<const Eigen::Matrix<T, -1, -1>> deformed(x, field_, num_nods_);
#pragma omp parallel for
    for (size_t cell_id = 0; cell_id < num_cells_; ++cell_id)
    {
        Matrix<T, field_, dim_>                                   def_gra;
        const Matrix<T, field_, num_per_cell_>                    x_cell      = indexing(deformed, field_all_rows_, cells_.col(cell_id));
        Matrix<T, field_ * num_per_cell_, 1>                      gra_x_based = Matrix<T, field_ * num_per_cell_, 1>::Zero();
        Matrix<T, field_ * dim_, field_ * dim_>                   hes_F_based;
        Matrix<T, field_ * num_per_cell_, field_ * num_per_cell_> hes_x_based;
        hes_x_based.setZero();

        //TODO:considering the order of basis
        for (size_t qdrt_id = 0; qdrt_id < num_qdrt_; ++qdrt_id)
        {
            basis::get_def_gra(Dphi_Dxi_[cell_id][qdrt_id], x_cell.data(), Dm_inv_[cell_id][qdrt_id], def_gra);
            hes_F_based = csttt::hes(def_gra, mtr_.col(cell_id));
            hes_x_based.noalias() += Ddef_Dx_[cell_id][qdrt_id].transpose() * hes_F_based * Ddef_Dx_[cell_id][qdrt_id] * quadrature_.WGT_[qdrt_id] * Jac_det_[cell_id][qdrt_id];
        }
        //save hes

        for (size_t p = 0; p < field_ * num_per_cell_; ++p)
        {
            for (size_t q = 0; q < field_ * num_per_cell_; ++q)
            {
                const size_t I = cells_(p / field_, cell_id) * field_ + p % field_;
                const size_t J = cells_(q / field_, cell_id) * field_ + q % field_;
                if (fabs(hes_x_based(p, q)) > 1e-10)
                    data->save_hes(I, J, hes_x_based(p, q));
            }
        }
    }
    return 0;
}

#define DECLARE_FEM_ELAS(FLOAT, CSTTT)                                       \
    template class                                                           \
        finite_element<FLOAT, 3, 3, 4, 1, 1, CSTTT, basis_func, quadrature>; \
    template class                                                           \
        finite_element<FLOAT, 3, 3, 8, 1, 2, CSTTT, basis_func, quadrature>;

#define DECLARE_FEM_ELAS_CSTTT(CSTTT) \
    DECLARE_FEM_ELAS(double, CSTTT)   \
    DECLARE_FEM_ELAS(float, CSTTT)

DECLARE_FEM_ELAS_CSTTT(linear_csttt);
DECLARE_FEM_ELAS_CSTTT(stvk);
DECLARE_FEM_ELAS_CSTTT(arap_csttt);
DECLARE_FEM_ELAS_CSTTT(corotated_csttt);

template class finite_element<double, 3, 1, 8, 1, 2, quadratic_csttt, basis_func, quadrature>;

}  // namespace PhysIKA
