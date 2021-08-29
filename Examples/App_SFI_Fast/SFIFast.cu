#include "SFIFast.h"
#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/Kernel.h"
#include "Dynamics/ParticleSystem/DensityPBD.h"
#include "Dynamics/ParticleSystem/ImplicitViscosity.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(SFIFast, TDataType)

template <typename TDataType>
SFIFast<TDataType>::SFIFast(std::string name)
    : Node(name)
{
    auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");

    m_position.connect(&pbf->m_position);
    m_vels.connect(&pbf->m_velocity);
    m_force.connect(&pbf->m_forceDensity);
}

template <typename TDataType>
void SFIFast<TDataType>::setInteractionDistance(Real d)
{
}

template <typename TDataType>
SFIFast<TDataType>::~SFIFast()
{
}

template <typename TDataType>
bool SFIFast<TDataType>::initialize()
{
    return true;
}

template <typename TDataType>
bool SFIFast<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
{
    return false;
}

template <typename TDataType>
bool SFIFast<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
{
    this->addChild(child);
    m_particleSystems.push_back(child);

    return false;
}

template <typename TDataType>
bool SFIFast<TDataType>::resetStatus()
{
    return true;
}

template <typename TDataType>
void SFIFast<TDataType>::advance(Real dt)
{
    int total_num = 0;
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        total_num += points.size();
    }

    if (m_position.getElementCount() != total_num)
    {
        m_position.setElementCount(total_num);
    }
    if (m_vels.getElementCount() != total_num)
    {
        m_vels.setElementCount(total_num);
    }
    if (m_force.getElementCount() != total_num)
    {
        m_force.setElementCount(total_num);
    }

    int                 start     = 0;
    DeviceArray<Coord>& allpoints = m_position.getValue();
    DeviceArray<Coord>& allvels   = m_vels.getValue();
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        DeviceArray<Coord>& vels   = m_particleSystems[i]->currentVelocity()->getValue();
        int                 num    = points.size();
        cudaMemcpy(allpoints.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        cudaMemcpy(allvels.begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        start += num;
    }

    auto nModel = this->getNumericalModel();
    nModel->step(this->getDt());

    start = 0;
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        DeviceArray<Coord>& vels   = m_particleSystems[i]->currentVelocity()->getValue();
        int                 num    = points.size();
        cudaMemcpy(points.begin(), allpoints.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        cudaMemcpy(vels.begin(), allvels.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

        start += num;
    }
}
}  // namespace PhysIKA