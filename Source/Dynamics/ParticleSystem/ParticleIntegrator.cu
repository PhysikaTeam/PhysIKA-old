#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "Framework/Framework/SceneGraph.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleIntegrator, TDataType)

template <typename TDataType>
ParticleIntegrator<TDataType>::ParticleIntegrator()
    : NumericalIntegrator()
{
}

template <typename TDataType>
void ParticleIntegrator<TDataType>::begin()
{
    if (!this->inPosition()->isEmpty())
    {
        int num = this->inPosition()->getElementCount();

        m_prePosition.resize(num);
        m_preVelocity.resize(num);

        Function1Pt::copy(m_prePosition, this->inPosition()->getValue());
        Function1Pt::copy(m_preVelocity, this->inVelocity()->getValue());

        this->inForceDensity()->getReference()->reset();
    }
}

template <typename TDataType>
void ParticleIntegrator<TDataType>::end()
{
}

template <typename TDataType>
bool ParticleIntegrator<TDataType>::initializeImpl()
{
    // 		if (!isAllFieldsReady())
    // 		{
    // 			std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
    // 			return false;
    // 		}
    //
    // 		int num = this->inPosition()->getElementCount();
    //
    // 		m_prePosition.resize(num);
    // 		m_preVelocity.resize(num);

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
bool ParticleIntegrator<TDataType>::updateVelocity()
{
    Real  dt      = getParent()->getDt();
    Coord gravity = SceneGraph::getInstance().getGravity();

    int total_num = this->inPosition()->getElementCount();
    cuExecute(total_num, K_UpdateVelocity, this->inVelocity()->getValue(), this->inForceDensity()->getValue(), gravity, dt);

    return true;
}

template <typename Real, typename Coord>
__global__ void K_UpdatePosition(
    DeviceArray<Coord> pos,
    DeviceArray<Coord> vel,
    Real               dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= pos.size())
        return;

    pos[pId] += dt * vel[pId];
}

template <typename TDataType>
bool ParticleIntegrator<TDataType>::updatePosition()
{
    Real dt = getParent()->getDt();

    int total_num = this->inPosition()->getReference()->size();
    cuExecute(total_num, K_UpdatePosition, this->inPosition()->getValue(), this->inVelocity()->getValue(), dt);

    return true;
}

template <typename TDataType>
bool ParticleIntegrator<TDataType>::integrate()
{
    if (!this->inPosition()->isEmpty())
    {
        updateVelocity();
        updatePosition();
    }

    return true;
}
}  // namespace PhysIKA