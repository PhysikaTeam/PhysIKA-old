/**
 * @author     : Chen Xiaosong (xiaosong0911@gmail.com)
 * @date       : 2019-06-13
 * @description: Implementation of MultipleFluidModel class, which implements the paper
 *               <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-27
 * @description: poslish code
 * @version    : 1.1
 */

#include "MultipleFluidModel.h"

#include "Core/Utility.h"
#include "Framework/ModuleTypes.h"
#include "DensityPBD.h"
#include "ParticleIntegrator.h"
#include "SummationDensity.h"
#include "ImplicitViscosity.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(MultipleFluidModel, TDataType)

template <typename Real, typename Coord, typename PhaseVector>
__global__ void UpdateMassInv(
    DeviceArray<Real>        massInvArr,
    DeviceArray<PhaseVector> cArr,
    PhaseVector              rho0)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= cArr.size())
        return;
    Real rho        = rho0.dot(cArr[pId]);
    massInvArr[pId] = Real(1) / rho;
}
template <typename Real, typename Coord, typename PhaseVector>
__global__ void UpdateColor(DeviceArray<Vector3f>    colorArr,
                            DeviceArray<PhaseVector> cArr)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= cArr.size())
        return;
    auto c        = cArr[pId];
    colorArr[pId] = { c[0], c[0], c[0] };
}

template <typename TDataType>
MultipleFluidModel<TDataType>::MultipleFluidModel()
    : NumericalModel()
{
    m_smoothingLength.setValue(Real(0.022));
    m_restDensity.setValue(PhaseVector(1, 5));

    attachField(&m_smoothingLength, "smoothingLength", "Smoothing length", false);

    attachField(&m_position, "position", "Storing the particle positions!", false);
    attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
    attachField(&m_massInv, "mass_inverse", "Storing mass inverse of particles!", false);
}

template <typename TDataType>
MultipleFluidModel<TDataType>::~MultipleFluidModel()
{
}

template <typename TDataType>
bool MultipleFluidModel<TDataType>::initializeImpl()
{
    m_concentration.setElementCount(m_position.getElementCount());
    m_massInv.setElementCount(m_position.getElementCount());
    m_color.setElementCount(m_position.getElementCount());
    int  num   = m_position.getElementCount();
    uint pDims = cudaGridSize(num, BLOCK_SIZE);

    // initialize random concentration
    std::vector<PhaseVector> cArr(num);
    for (auto& c : cArr)
    {
        c[0] = Real(0.5) + ((Real(rand()) / RAND_MAX) * 2 - 1) * Real(0.05);
        c[1] = 1 - c[0];
    }
    Function1Pt::copy(m_concentration.getValue(), cArr);

    // Create modules
    m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
    m_smoothingLength.connect(m_nbrQuery->inRadius());
    m_position.connect(m_nbrQuery->inPosition());
    m_nbrQuery->initialize();

    m_pbdModule = std::make_shared<DensityPBD<TDataType>>();
    m_smoothingLength.connect(m_pbdModule->varSmoothingLength());
    m_position.connect(m_pbdModule->inPosition());
    m_velocity.connect(m_pbdModule->inVelocity());
    m_massInv.connect(&m_pbdModule->m_massInv);
    m_nbrQuery->outNeighborhood()->connect(m_pbdModule->inNeighborIndex());
    m_pbdModule->initialize();

    m_phaseSolver = std::make_shared<CahnHilliard<TDataType>>();
    m_position.connect(&m_phaseSolver->m_position);
    m_concentration.connect(&m_phaseSolver->m_concentration);
    m_nbrQuery->outNeighborhood()->connect(&m_phaseSolver->m_neighborhood);
    m_smoothingLength.connect(&m_phaseSolver->m_smoothingLength);
    m_phaseSolver->initialize();

    m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
    m_position.connect(m_integrator->inPosition());
    m_velocity.connect(m_integrator->inVelocity());
    m_forceDensity.connect(m_integrator->inForceDensity());
    m_integrator->initialize();

    m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
    m_visModule->setViscosity(Real(1));
    m_smoothingLength.connect(&m_visModule->m_smoothingLength);
    m_position.connect(&m_visModule->m_position);
    m_velocity.connect(&m_visModule->m_velocity);
    m_nbrQuery->outNeighborhood()->connect(&m_visModule->m_neighborhood);
    m_visModule->initialize();

    Node* parent = this->getParent();
    m_nbrQuery->setParent(parent);
    m_integrator->setParent(parent);
    m_phaseSolver->setParent(parent);
    m_pbdModule->setParent(parent);
    m_visModule->setParent(parent);

    return NumericalModel::initializeImpl();
}

template <typename TDataType>
void MultipleFluidModel<TDataType>::step(Real dt)
{
    int  num   = m_position.getElementCount();
    uint pDims = cudaGridSize(num, BLOCK_SIZE);
    m_integrator->begin();

    m_nbrQuery->compute();

    m_integrator->integrate();

    m_phaseSolver->integrate();

    UpdateMassInv<Real, Coord, PhaseVector><<<pDims, BLOCK_SIZE>>>(
        m_massInv.getValue(), m_concentration.getValue(), m_restDensity.getValue());
    m_pbdModule->constrain();

    m_visModule->constrain();

    m_integrator->end();

    UpdateColor<Real, Coord><<<pDims, BLOCK_SIZE>>>(
        m_color.getValue(), m_concentration.getValue());
}
}  // namespace PhysIKA