/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: solver list helper
 * @version    : 1.0
 */
#ifndef PhysIKA_SOLVER_LIST
#define PhysIKA_SOLVER_LIST
#include "Solver/linear_solver/pcg.h"
#include "Solver/linear_solver/gpu_pcg.cuh"
#include "coro_solver.h"
#include "fast_ms_solver.h"

namespace PhysIKA {

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_pcg(
    const std::shared_ptr<Problem<T, dim>>& pb,
    const boost::property_tree::ptree&      pt,
    std::shared_ptr<dat_str_core<T, dim>>   dat_str)
{

    EIGEN_PCG<T>          pcg(pt.get<bool>("hes_is_const", false), pt.get<T>("cg_tol", 1e-3));
    linear_solver_type<T> LS = std::bind(&EIGEN_PCG<T>::solve, pcg, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    return std::make_shared<newton_base<T, dim>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-4), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), LS, dat_str);
}

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_fast_ms_and_embedded(
    const std::shared_ptr<Problem<T, dim>>&  pb,
    const boost::property_tree::ptree&       pt,
    std::shared_ptr<dat_str_core<T, dim>>    dat_str,
    size_t                                   dof_of_nods,
    std::shared_ptr<embedded_interpolate<T>> embedded_interp,
    std::shared_ptr<semi_implicit<T>>        semi,
    std::shared_ptr<fast_ms_info<T>>         solver_info)
{

    EIGEN_PCG<T>          pcg(true, pt.get<T>("cg_tol", 1e-3));
    linear_solver_type<T> LS = std::bind(&EIGEN_PCG<T>::solve, pcg, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    return std::make_shared<fast_ms_solver<T, dim>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-4), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), LS, dat_str, dof_of_nods, embedded_interp, semi, solver_info);
}

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_pcg_and_embedded(
    const std::shared_ptr<Problem<T, dim>>&  pb,
    const boost::property_tree::ptree&       pt,
    std::shared_ptr<dat_str_core<T, dim>>    dat_str,
    size_t                                   dof_of_nods,
    std::shared_ptr<embedded_interpolate<T>> embedded_interp,
    std::shared_ptr<semi_implicit<T>>        semi = nullptr)
{

    EIGEN_PCG<T>          pcg(true, pt.get<T>("cg_tol", 1e-3));
    linear_solver_type<T> LS = std::bind(&EIGEN_PCG<T>::solve, pcg, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    return std::make_shared<newton_base_with_embedded<T, dim>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-4), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), LS, dat_str, dof_of_nods, embedded_interp, semi);
}

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_gpu_pcg(
    const std::shared_ptr<Problem<T, dim>>& pb,
    const boost::property_tree::ptree&      pt,
    std::shared_ptr<dat_str_core<T, dim>>   dat_str)
{
    CUDA_PCG<T>           pcg(true);
    linear_solver_type<T> LS = std::bind(&CUDA_PCG<T>::solve, pcg, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    return std::make_shared<newton_base<T, dim>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-4), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), LS, dat_str);
}

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_gmg_coro_pcg(
    const std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>>>& fac_hes_rest,
    const SPM_R<T>&                                                       hes_rest,
    const std::shared_ptr<Problem<T, dim>>&                               pb,
    const boost::property_tree::ptree&                                    pt,
    std::shared_ptr<elas_intf<T, 3>>&                                     elas_intf_ptr,
    std::shared_ptr<dat_str_core<T, dim>>                                 dat_str)
{
    return std::make_shared<gmg_coro_solver<T>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-4), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), fac_hes_rest, elas_intf_ptr, hes_rest, pt.get<T>("cg_tol", 1e-10), dat_str);
}

template <typename T, size_t dim>
std::shared_ptr<newton_base<T, dim>> newton_with_amg_coro_pcg(
    const std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>>>& fac_hes_rest,
    const std::shared_ptr<Problem<T, dim>>&                               pb,
    const boost::property_tree::ptree&                                    pt,
    const SPM_R<T>&                                                       hes_rest,
    std::shared_ptr<dat_str_core<T, dim>>                                 dat_str)
{
    return std::make_shared<amg_coro_solver<T>>(pb, pt.get<size_t>("newton_MaxIter", 20), pt.get<T>("newton_tol", 1e-2), pt.get<bool>("line_search", false), pt.get<bool>("hes_is_const", false), hes_rest, fac_hes_rest, pt.get<T>("cg_tol", 1e-10), dat_str);
}

};  // namespace PhysIKA
#endif
