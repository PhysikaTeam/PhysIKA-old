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
PhysIKA::SFIFast<TDataType>::SFIFast(std::string name)
    : Node(name)
{
    this->attachField(&radius, "radius", "radius");
    radius.setValue(0.0075);

    m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
    radius.connect(m_nbrQuery->inRadius());
    m_position.connect(m_nbrQuery->inPosition());

    m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("collision");
    radius.connect(m_pbdModule->varSmoothingLength());
    m_position.connect(m_pbdModule->inPosition());
    m_vels.connect(m_pbdModule->inVelocity());
    m_nbrQuery->outNeighborhood()->connect(m_pbdModule->inNeighborIndex());
    m_pbdModule->varIterationNumber()->setValue(5);

    m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
    m_visModule->setViscosity(Real(1));
    radius.connect(&m_visModule->m_smoothingLength);
    m_position.connect(&m_visModule->m_position);
    m_vels.connect(&m_visModule->m_velocity);
    m_nbrQuery->outNeighborhood()->connect(&m_visModule->m_neighborhood);
    m_visModule->initialize();
}

template <typename TDataType>
void SFIFast<TDataType>::setInteractionDistance(Real d)
{
    radius.setValue(d);
    m_pbdModule->varSamplingDistance()->setValue(d / 2);
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
    int               total_num = 0;
    std::vector<int>  ids;
    std::vector<Real> mass;
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        auto points = m_particleSystems[i]->currentPosition()->getValue();
        total_num += points.size();
        Real m = m_particleSystems[i]->getMass() / points.size();
        for (int j = 0; j < points.size(); j++)
        {
            ids.push_back(i);
            mass.push_back(m);
        }
    }

    m_objId.resize(total_num);
    m_vels.setElementCount(total_num);
    m_mass.setElementCount(total_num);
    m_position.setElementCount(total_num);

    posBuf.resize(total_num);
    weights.resize(total_num);
    init_pos.resize(total_num);

    velBuf.resize(total_num);
    velOld.resize(total_num);

    Function1Pt::copy(m_objId, ids);
    Function1Pt::copy(m_mass.getValue(), mass);
    ids.clear();
    mass.clear();

    int                 start     = 0;
    DeviceArray<Coord>& allpoints = m_position.getValue();
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        DeviceArray<Coord>& vels   = m_particleSystems[i]->currentVelocity()->getValue();
        int                 num    = points.size();
        cudaMemcpy(allpoints.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        cudaMemcpy(m_vels.getValue().getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        start += num;
    }

    return true;
}

template <typename Real, typename Coord>
__global__ void K_Collide(
    DeviceArray<int>   objIds,
    DeviceArray<Real>  mass,
    DeviceArray<Coord> points,
    DeviceArray<Coord> newPoints,
    DeviceArray<Real>  weights,
    NeighborList<int>  neighbors,
    Real               radius)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= points.size())
        return;

    SpikyKernel<Real> kernel;

    Real  r;
    Coord pos_i   = points[pId];
    int   id_i    = objIds[pId];
    Real  mass_i  = 1.0f;  //mass[pId];
    int   nbSize  = neighbors.getNeighborSize(pId);
    int   col_num = 0;
    Coord pos_num = Coord(0);
    for (int ne = 0; ne < nbSize; ne++)
    {
        int   j     = neighbors.getElement(pId, ne);
        Coord pos_j = points[j];

        r = (pos_i - pos_j).norm();
        if (r < radius && objIds[j] != id_i)
        {
            col_num++;
            Real mass_j = 1.0f;
            //mass[j];
            Coord center = (pos_i + pos_j) / 2;
            Coord n      = pos_i - pos_j;
            n            = n.norm() < EPSILON ? Coord(0, 0, 0) : n.normalize();

            Real a = mass_i / (mass_i + mass_j);

            Real d = radius - r;

            Coord target_i = pos_i + (1 - a) * d * n;  // (center + 0.5*radius*n);
            Coord target_j = pos_j - a * d * n;        // (center - 0.5*radius*n);
            //				pos_num += (center + 0.4*radius*n);

            Real weight = kernel.Weight(r, 2 * radius);

            atomicAdd(&newPoints[pId][0], weight * target_i[0]);
            atomicAdd(&newPoints[j][0], weight * target_j[0]);

            atomicAdd(&weights[pId], weight);
            atomicAdd(&weights[j], weight);

            if (Coord::dims() >= 2)
            {
                atomicAdd(&newPoints[pId][1], weight * target_i[1]);
                atomicAdd(&newPoints[j][1], weight * target_j[1]);
            }

            if (Coord::dims() >= 3)
            {
                atomicAdd(&newPoints[pId][2], weight * target_i[2]);
                atomicAdd(&newPoints[j][2], weight * target_j[2]);
            }
        }
    }

    
}

template <typename Real, typename Coord>
__global__ void K_Viscosity(
    DeviceArray<int>   objIds,
    DeviceArray<Real>  mass,
    DeviceArray<Coord> points,
    DeviceArray<Coord> oldVels,
    DeviceArray<Coord> newVels,
    DeviceArray<Real>  weights,
    NeighborList<int>  neighbors,
    Real               radius)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= points.size())
        return;

    if (objIds[pId] == 0)
        return;

    SpikyKernel<Real> kernel;

    Real  r;
    Coord pos_i   = points[pId];
    int   id_i    = objIds[pId];
    Real  mass_i  = mass[pId];
    int   nbSize  = neighbors.getNeighborSize(pId);
    int   col_num = 0;
    Coord pos_num = Coord(0);
    for (int ne = 0; ne < nbSize; ne++)
    {
        int   j     = neighbors.getElement(pId, ne);
        Coord pos_j = points[j];

        r = (pos_i - pos_j).norm();
        if (r < radius)
        {
            Real mass_j = mass[j];

            Real weight = kernel.Weight(r, 2 * radius) * mass_j;

            Coord vel_j = oldVels[j] * weight;
            atomicAdd(&weights[pId], weight);
            atomicAdd(&newVels[pId][0], vel_j[0]);
            atomicAdd(&newVels[pId][1], vel_j[1]);
            atomicAdd(&newVels[pId][2], vel_j[2]);
        }
    }
   /* if (newVels[pId].norm() > EPSILON)
        printf("!!!!!!!!!! %.10lf  %.10lf  %.10lf\n", 
               newVels[pId][0] / weights[pId],
               newVels[pId][1] / weights[pId],
               newVels[pId][2] / weights[pId]);*/

    
}

template <typename Real, typename Coord>
__global__ void K_ComputeTarget(
    DeviceArray<Coord> oldPoints,
    DeviceArray<Coord> newPoints,
    DeviceArray<Real>  weights)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= oldPoints.size())
        return;

    if (weights[pId] > EPSILON)
    {
        newPoints[pId] /= weights[pId];
    }
    else
        newPoints[pId] = oldPoints[pId];
}

template <typename Real, typename Coord>
__global__ void K_ComputeVelocity(
    DeviceArray<Coord> initPoints,
    DeviceArray<Coord> curPoints,
    DeviceArray<Coord> velocites,
    Real               dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velocites.size())
        return;

    velocites[pId] += 0.5 * (curPoints[pId] - initPoints[pId]) / dt;
}


template <typename Coord>
__global__ void SFIF_UpdateViscosity(
    DeviceArray<Coord> velocities,
    DeviceArray<Coord> velocity_old,
    DeviceArray<Coord> velocity_new
)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velocities.size())
        return;

    velocities[pId] = (0.65f * velocity_old[pId] + 0.35f * velocity_new[pId]);
    /*if (velocity_new[pId].norm() > EPSILON)
        printf("YES  %.3lf   %.3lf  %.3lf\n", velocity_new[pId][0], velocity_new[pId][1], velocity_new[pId][2]);*/
}

template <typename TDataType>
void SFIFast<TDataType>::advance(Real dt)
{
    int                 start     = 0;
    DeviceArray<Coord>& allpoints = m_position.getValue();
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        DeviceArray<Coord>& vels   = m_particleSystems[i]->currentVelocity()->getValue();
        int                 num    = points.size();
        cudaMemcpy(allpoints.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        cudaMemcpy(m_vels.getValue().getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        start += num;
    }

    m_nbrQuery->compute();

   

    auto module2 = this->template getModule<ImplicitViscosity<TDataType>>("viscosity");
    //module2->constrain();

    Function1Pt::copy(init_pos, allpoints);

	uint pDims = cudaGridSize(allpoints.size(), BLOCK_SIZE);
	for (size_t it = 0; it < 5; it++)
	{
		weights.reset();
		posBuf.reset();
		K_Collide << <pDims, BLOCK_SIZE >> > (
			m_objId, 
			m_mass.getValue(),
			allpoints,
			posBuf, 
			weights, 
			m_nbrQuery->outNeighborhood()->getValue(),
			radius.getValue());

		K_ComputeTarget << <pDims, BLOCK_SIZE >> > (
			allpoints,
			posBuf, 
			weights);

		Function1Pt::copy(allpoints, posBuf);
	}

	K_ComputeVelocity << <pDims, BLOCK_SIZE >> > (init_pos, allpoints, m_vels.getValue(), getParent()->getDt());

    Function1Pt::copy(velOld, m_vels.getValue());
    for (size_t it = 0; it < 3; it++)
    {
        weights.reset();
        velBuf.reset();
        K_Viscosity<<<pDims, BLOCK_SIZE>>>(
            m_objId,
            m_mass.getValue(),
            allpoints,
            m_vels.getValue(),
            velBuf,
            weights,
            m_nbrQuery->outNeighborhood()->getValue(),
            radius.getValue());

        K_ComputeTarget<<<pDims, BLOCK_SIZE>>>(
            m_vels.getValue(),
            velBuf,
            weights);

        Function1Pt::copy(m_vels.getValue(), velBuf);
    }
    //Function1Pt::copy(velBuf, m_vels.getValue());
    SFIF_UpdateViscosity<<<pDims, BLOCK_SIZE>>>(m_vels.getValue(), velOld, velBuf);

    /*auto module = this->template getModule<DensityPBD<TDataType>>("collision");
    module->constrain();*/

    start = 0;
    for (int i = 0; i < m_particleSystems.size(); i++)
    {
        DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
        DeviceArray<Coord>& vels   = m_particleSystems[i]->currentVelocity()->getValue();
        int                 num    = points.size();
        cudaMemcpy(points.getDataPtr(), allpoints.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
        cudaMemcpy(vels.getDataPtr(), m_vels.getValue().getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

        start += num;
    }
}
}  // namespace PhysIKA