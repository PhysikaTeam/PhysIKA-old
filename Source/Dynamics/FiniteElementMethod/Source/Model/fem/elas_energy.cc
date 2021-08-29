/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity energy implementation for finite element method.
 * @version    : 1.0
 */
#include "Common/polar_decomposition.h"
#include "elas_energy.h"

using namespace std;
using namespace Eigen;
namespace PhysIKA {

ELAS_TEMP
ELAS_CLASS::BaseElas(const Matrix<T, dim_, -1>& nods, const Matrix<int, num_per_cell_, -1>& cells, const T& ym, const T& poi)
    : base_class(nods, cells)
{
    T mu, lambda;
    compute_lame_coeffs(ym, poi, mu, lambda);
    base_class::mtr_.resize(2, cells.cols());
    base_class::mtr_.row(0) = Matrix<T, 1, -1>::Ones(cells.cols()) * lambda;
    base_class::mtr_.row(1) = Matrix<T, 1, -1>::Ones(cells.cols()) * mu;
}

ELAS_TEMP
ELAS_CLASS::BaseElas(const Matrix<T, dim_, -1>& nods, const Matrix<int, num_per_cell_, -1>& cells, const VEC<T>& ym, const VEC<T>& poi)
    : base_class(nods, cells)
{

    base_class::mtr_.resize(2, cells.cols());
#pragma omp parallel for
    for (size_t i = 0; i < cells.cols(); ++i)
    {
        T mu, lambda;
        compute_lame_coeffs(ym(i), poi(i), mu, lambda);
        this->mtr_(0, i) = lambda;
        this->mtr_(1, i) = mu;
    }
}

ELAS_TEMP
int ELAS_CLASS::aver_ele_R(const T* x, vec_mat& vec_R) const
{
    vec_R.resize(this->num_nods_);
    vec_R.assign(this->num_nods_, Matrix<T, dim_, dim_>::Zero());

    Map<const Matrix<T, -1, -1>> deformed(x, dim_, this->num_nods_);
    vector<size_t>               num_nei(this->num_nods_, 0);
//calc R for each cell
#pragma omp parallel for
    for (size_t cell_id = 0; cell_id < this->num_cells_; ++cell_id)
    {
        Matrix<T, dim_, dim_>                def_gra;
        const Matrix<T, dim_, num_per_cell_> x_cell = indexing(deformed, this->field_all_rows_, this->cells_.col(cell_id));
        Matrix<T, dim_, dim_>                R_cell = Matrix<T, dim_, dim_>::Zero();
        for (size_t qdrt_id = 0; qdrt_id < this->num_qdrt_; ++qdrt_id)
        {
            base_class::basis::get_def_gra(this->Dphi_Dxi_[cell_id][qdrt_id], x_cell.data(), this->Dm_inv_[cell_id][qdrt_id], def_gra);
            Matrix<T, dim_, dim_> R_qdrt;
            polar_decomposition<T, Matrix<T, dim_, dim_>>(def_gra, R_qdrt);
            R_cell += R_qdrt;
        }
        Matrix<T, dim_, dim_> R_cell_corr = Matrix<T, dim_, dim_>::Zero();
        polar_decomposition<T, Matrix<T, dim_, dim_>>(R_cell, R_cell_corr);

        for (size_t i = 0; i < num_per_cell_; ++i)
        {
            const size_t nod_id = this->cells_(i, cell_id);
            for (size_t m = 0; m < 3; ++m)
                for (size_t n = 0; n < 3; ++n)
                {
#pragma omp atomic
                    vec_R[nod_id](m, n) += R_cell_corr(m, n);
                }
#pragma omp atomic
            num_nei[nod_id] += 1;
        }
    }

//calc R for each nod
#pragma omp parallel for
    for (size_t nod_id = 0; nod_id < this->num_nods_; ++nod_id)
    {
        Matrix<T, dim_, dim_> R_nod = Matrix<T, dim_, dim_>::Zero();
        vec_R[nod_id] /= num_nei[nod_id];
        polar_decomposition<T, Matrix<T, dim_, dim_>>(vec_R[nod_id], R_nod);
        vec_R[nod_id] = R_nod;
    }
    return 0;
}

#define DECLARE_BaseElas(FLOAT, CSTTT)                              \
    template class                                                  \
        BaseElas<FLOAT, 3, 4, 1, 1, CSTTT, basis_func, quadrature>; \
    template class                                                  \
        BaseElas<FLOAT, 3, 8, 1, 2, CSTTT, basis_func, quadrature>;

#define DECLARE_BaseElas_CSTTT(CSTTT) DECLARE_BaseElas(double, CSTTT) \
    DECLARE_BaseElas(float, CSTTT)

DECLARE_BaseElas_CSTTT(linear_csttt)
    DECLARE_BaseElas_CSTTT(stvk)
        DECLARE_BaseElas_CSTTT(arap_csttt)
            DECLARE_BaseElas_CSTTT(corotated_csttt)

                template <typename T, typename ELAS_TYPE, size_t dim_, size_t num_per_cell_>
                inline int gen_elas_energy_intf_for_one_type(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const T& ym, const T& poi, std::shared_ptr<Functional<T, dim_>>& energy, std::shared_ptr<elas_intf<T, dim_>>* intf)
{
    std::shared_ptr<ELAS_TYPE> elas = std::make_shared<ELAS_TYPE>(nods, cells, ym, poi);
    energy                          = dynamic_pointer_cast<Functional<T, dim_>>(elas);
    if (intf != nullptr)
        *intf = dynamic_pointer_cast<elas_intf<T, dim_>>(elas);
    return 0;
}

template <typename T, typename ELAS_TYPE, size_t dim_, size_t num_per_cell_>
inline int gen_elas_energy_intf_for_one_type(const Eigen::Matrix<T, dim_, -1>& nods, const Eigen::Matrix<int, num_per_cell_, -1>& cells, const Ref<VEC<T>>& ym, const Ref<VEC<T>>& poi, std::shared_ptr<Functional<T, dim_>>& energy, std::shared_ptr<elas_intf<T, dim_>>* intf)
{
    std::shared_ptr<ELAS_TYPE> elas = std::make_shared<ELAS_TYPE>(nods, cells, ym, poi);
    energy                          = dynamic_pointer_cast<Functional<T, dim_>>(elas);
    if (intf != nullptr)
        *intf = dynamic_pointer_cast<elas_intf<T, dim_>>(elas);
    return 0;
}

template <typename T>
int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<T, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const T& Young, const T& poi, std::shared_ptr<Functional<T, 3>>& energy, std::shared_ptr<elas_intf<T, 3>>* intf)
{
    if (type == "tet")
    {
        if (csttt_type == "linear")
            gen_elas_energy_intf_for_one_type<T, TET_lin_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "stvk")
            gen_elas_energy_intf_for_one_type<T, TET_stvk_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "corotated")
            gen_elas_energy_intf_for_one_type<T, TET_corotated_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "arap")
            gen_elas_energy_intf_for_one_type<T, TET_arap_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else
        {
            cout << "Error: constutive type should be linear or stvk." << endl;
            return __LINE__;
        }
    }
    else
    {
        if (csttt_type == "linear")
            gen_elas_energy_intf_for_one_type<T, HEX_lin_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "stvk")
            gen_elas_energy_intf_for_one_type<T, HEX_stvk_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "corotated")
            gen_elas_energy_intf_for_one_type<T, HEX_corotated_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "arap")
            gen_elas_energy_intf_for_one_type<T, HEX_arap_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else
        {
            cout << "Error: constutive type should be linear or stvk." << endl;
            return __LINE__;
        }
    }
}

template int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<double, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const double& Young, const double& poi, std::shared_ptr<Functional<double, 3>>& energy, std::shared_ptr<elas_intf<double, 3>>* intf);
template int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<float, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const float& Young, const float& poi, std::shared_ptr<Functional<float, 3>>& energy, std::shared_ptr<elas_intf<float, 3>>* intf);

template <typename T>
int gen_elas_energy_intf(const string& type, const string& csttt_type, const Matrix<T, 3, -1>& nods, const Matrix<int, -1, -1>& cells, const Ref<VEC<T>>& Young, const Ref<VEC<T>>& poi, shared_ptr<Functional<T, 3>>& energy, shared_ptr<elas_intf<T, 3>>* intf)
{
    if (type == "tet")
    {
        if (csttt_type == "linear")
            return gen_elas_energy_intf_for_one_type<T, TET_lin_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "stvk")
            return gen_elas_energy_intf_for_one_type<T, TET_stvk_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "corotated")
            return gen_elas_energy_intf_for_one_type<T, TET_corotated_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "arap")
            return gen_elas_energy_intf_for_one_type<T, TET_arap_ELAS<T>, 3, 4>(nods, cells, Young, poi, energy, intf);
        else
        {
            cout << "Error: constutive type should be linear or stvk." << endl;
            return __LINE__;
        }
    }
    else
    {
        if (csttt_type == "linear")
            return gen_elas_energy_intf_for_one_type<T, HEX_lin_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "stvk")
            return gen_elas_energy_intf_for_one_type<T, HEX_stvk_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "corotated")
            return gen_elas_energy_intf_for_one_type<T, HEX_corotated_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else if (csttt_type == "arap")
            return gen_elas_energy_intf_for_one_type<T, HEX_arap_ELAS<T>, 3, 8>(nods, cells, Young, poi, energy, intf);
        else
        {
            cout << "Error: constutive type should be linear or stvk." << endl;
            return __LINE__;
        }
    }
}

template int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<double, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const Ref<VEC<double>>& Young, const Ref<VEC<double>>& poi, std::shared_ptr<Functional<double, 3>>& energy, std::shared_ptr<elas_intf<double, 3>>* intf);
template int gen_elas_energy_intf(const std::string& type, const std::string& csttt_type, const Eigen::Matrix<float, 3, -1>& nods, const Eigen::Matrix<int, -1, -1>& cells, const Ref<VEC<float>>& Young, const Ref<VEC<float>>& poi, std::shared_ptr<Functional<float, 3>>& energy, std::shared_ptr<elas_intf<float, 3>>* intf);

}  // namespace PhysIKA
