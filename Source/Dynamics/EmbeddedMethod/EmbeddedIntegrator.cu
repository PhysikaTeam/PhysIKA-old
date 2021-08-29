/**
 * @author     : ZHAO CHONGYAO (cyzhao@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: A implicit integrator source for physika library
 * @version    : 2.2.1
 */
#include <iostream>
#include <cuda_runtime.h>

#include "Framework/Framework/FieldArray.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/SceneGraph.h"
#include "Problem/integrated_problem/embedded_elas_fem_problem.h"
#include "Problem/integrated_problem/fast_ms_problem.h"
#include "Common/data_str_core.h"
#include "Solver/solver_lists.h"
#include "EmbeddedIntegrator.h"

using namespace std;
namespace PhysIKA {
template <typename TDataType>
EmbeddedIntegrator<TDataType>::EmbeddedIntegrator()
    : NumericalIntegrator()
{
    /*	attachField(&inPosition, "position", "Storing the particle positions!", false);
		attachField(&inVelocity, "velocity", "Storing the particle velocities!", false);
		attachField(&inForceDensity, "force", "Particle forces", false);*/
}

template <typename TDataType>
void EmbeddedIntegrator<TDataType>::begin()
{
    Function1Pt::copy(m_prePosition, this->inPosition()->getValue());
    Function1Pt::copy(m_preVelocity, this->inVelocity()->getValue());
    this->inForceDensity()->getReference()->reset();

    //========see velo=====//
    // static size_t  cnt = 0;

    // HostArray<Coord> m_velo_host(this->inVelocity()->getElementCount());
    // Function1Pt::copy(m_velo_host, this->inVelocity()->getValue());
    // std::cout << m_velo_host[0][1] << std::endl;
    // ++cnt;
    // if(cnt == 2)
    //   exit(EXIT_FAILURE);

    //========see velo=====//

    const size_t num = this->inPosition()->getElementCount();

    m_position_host.resize(num);
    Function1Pt::copy(m_position_host, this->inPosition()->getValue());
    pos_.resize(num * 3);
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i)
        for (size_t j = 0; j < 3; ++j)
            pos_[i * 3 + j] = m_position_host[i][j];

    m_velocity_host.resize(num);
    Function1Pt::copy(m_velocity_host, this->inVelocity()->getValue());
    vel_.resize(num * 3);
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i)
        for (size_t j = 0; j < 3; ++j)
            vel_[i * 3 + j] = m_velocity_host[i][j];

    epb_fac_->update_problem(&pos_[0]);
}

template <typename TDataType>
void EmbeddedIntegrator<TDataType>::end()
{
}

template <typename TDataType>
bool EmbeddedIntegrator<TDataType>::initializeImpl()
{
    if (!isAllFieldsReady())
    {
        std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
        return false;
    }

    int num = this->inPosition()->getElementCount();

    m_prePosition.resize(num);
    m_preVelocity.resize(num);

    return true;
}

template <typename Real, typename Coord>
__global__ void K_UpdateVelocity(
    DeviceArray<Coord> vel,
    DeviceArray<Coord> forceDensity,
    Coord              gravity,
    Real               dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= forceDensity.size())
        return;

    vel[pId] += dt * (forceDensity[pId] + gravity);
}

template <typename Real, typename Coord>
__global__ void K_UpdateVelocity(
    DeviceArray<Coord> vel,
    DeviceArray<Coord> pos,
    DeviceArray<Coord> pre_pos,
    Real               dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= pos.size())
        return;

    vel[pId] = (pos[pId] - pre_pos[pId]) / dt;
}

template <typename Real, typename Coord>
__global__ void K_UpdateVelocity(
    DeviceArray<Coord> vel,
    DeviceArray<Coord> force,
    DeviceArray<Real>  mass,
    Real               dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= force.size())
        return;

    vel[pId] += dt * force[pId] / mass[pId];
}

template <typename TDataType>
bool EmbeddedIntegrator<TDataType>::updateVelocity()
{
    Real  dt    = getParent()->getDt();
    cuint pDims = cudaGridSize(this->inPosition()->getReference()->size(), BLOCK_SIZE);
    K_UpdateVelocity<<<pDims, BLOCK_SIZE>>>(
        this->inVelocity()->getValue(),
        this->inPosition()->getValue(),
        m_prePosition,
        dt);

    return true;
}

template <typename TDataType>
bool EmbeddedIntegrator<TDataType>::updatePosition()
{
    auto pb = epb_fac_->build_problem();

    if (solver_type_ == "explicit")
    {
        for (int i = 0; i < 100; ++i)
        {
            pb      = epb_fac_->build_problem();
            solver_ = newton_with_pcg_and_embedded<Real, 3>(pb, pt_, dat_str_, pos_.size(), epb_fac_->get_embedded_interpolate(), semi_implicit_);
            solver_->solve(&pos_[0]);
            epb_fac_->update_problem(&pos_[0]);
        }
    }
    else if (solver_type_ == "fast_ms")
    {
        solver_ = newton_with_fast_ms_and_embedded<Real, 3>(pb, pt_, dat_str_, pos_.size(), epb_fac_->get_embedded_interpolate(), semi_implicit_, fast_ms_solver_info_);
    }
    else
        solver_ = newton_with_pcg_and_embedded<Real, 3>(pb, pt_, dat_str_, pos_.size(), epb_fac_->get_embedded_interpolate());

    solver_->solve(&pos_[0]);

    const size_t num = this->inPosition()->getElementCount();
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i)
        for (size_t j = 0; j < 3; ++j)
            m_position_host[i][j] = pos_[i * 3 + j];
    Function1Pt::copy(this->inPosition()->getValue(), m_position_host);

    return true;
}

template <typename TDataType>
bool EmbeddedIntegrator<TDataType>::integrate()
{
    updatePosition();
    updateVelocity();

    return true;
}

template <typename TDataType>
void EmbeddedIntegrator<TDataType>::bind_problem(const std::shared_ptr<embedded_problem_builder<Real, 3>>& epb_fac, const boost::property_tree::ptree& pt)
{
    solver_type_ = pt.get<string>("solver_type", "implicit");
    if (solver_type_ == "explicit")
    {
        embedded_interp_ = epb_fac->get_embedded_interpolate();
        semi_implicit_   = epb_fac->get_semi_implicit();

        Eigen::Matrix<Real, -1, -1>            nods        = epb_fac->get_nods();
        Eigen::Matrix<Real, -1, -1>            nods_coarse = nods * embedded_interp_->get_fine_to_coarse_coefficient();
        Eigen::Map<Eigen::Matrix<Real, -1, 1>> init_nods_coarse(nods_coarse.data(), nods_coarse.size());
        semi_implicit_->update_status(init_nods_coarse);
    }
    else if (solver_type_ == "fast_ms")
    {
        shared_ptr<fast_ms_builder<Real>> fast_ms_epb_fac = dynamic_pointer_cast<fast_ms_builder<Real>>(epb_fac);
        fast_ms_solver_info_                              = fast_ms_epb_fac->get_fast_ms_solver_info();
    }

    auto pb  = epb_fac->build_problem();
    dat_str_ = make_shared<dat_str_core<Real, 3>>(pb->Nx() / 3, pt.get<bool>("hes_is_const", false));
    exit_if(compute_hes_pattern(pb->energy_, dat_str_), "compute hes pattern fail");
    epb_fac_ = epb_fac;
    pt_      = pt;
}

}  // namespace PhysIKA
