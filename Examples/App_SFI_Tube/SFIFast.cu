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
void SFIFast<TDataType>::output_initialized_particles()
{
    HostArray<Coord> host_particle_positions;
    host_particle_positions.resize(m_position.getElementCount());
    Function1Pt::copy(host_particle_positions, m_position.getValue());
    std::ofstream fout;
    fout.open("data_fluid_pos.obj");

    for (int i = 0; i < host_particle_positions.size(); i++)
    {
        fout << "v " << host_particle_positions[i][0] << ' '
             << host_particle_positions[i][1] << ' '
             << host_particle_positions[i][2]
             << std::endl;
    }

    fout.close();

    host_particle_positions.release();
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
        Real m = m_particleSystems[i]->getMass();
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
            Real mass_j = 1.0f;  //mass[j];

            /* if (objIds[pId] == 0)
            {
                printf("%.10lf %.10lf \n", mass_i, mass_j);
            }*/
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

            Real weight = 1.0f * mass_j;

            Coord vel_j = oldVels[j] * weight;
            atomicAdd(&weights[pId], weight);
            atomicAdd(&newVels[pId][0], vel_j[0]);
            atomicAdd(&newVels[pId][1], vel_j[1]);
            atomicAdd(&newVels[pId][2], vel_j[2]);
        }
    }
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
    DeviceArray<Coord> velocity_new)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velocities.size())
        return;

    velocities[pId] = (0.0f * velocity_old[pId] + 1.0f * velocity_new[pId]);
    /*if (velocity_new[pId].norm() > EPSILON)
        printf("YES  %.3lf   %.3lf  %.3lf\n", velocity_new[pId][0], velocity_new[pId][1], velocity_new[pId][2]);*/
}

template <typename Coord>
__global__ void SFIF_UpdateBoundary_Tube(
    DeviceArray<Coord> velocities,
    DeviceArray<Coord> pos)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velocities.size())
        return;

    if (pos[pId][0] >= 2.0f)
    {
        Coord pos_i = pos[pId];
        if (pos[pId][2] < 1 / 6.0f)
        {
            Real left   = pos[pId][0] - 2.0f;
            Real behind = 1.0f / 6.0f - pos[pId][2];

            if (left < behind)
            {
                pos_i[0] = 2.0f;
            }
            else
            {
                pos_i[2] = 1.0f / 6.0f;
            }
        }
        else if (pos[pId][2] < 2.0f / 3.0f && pos[pId][2] > 1.0f / 3.0f)
        {
            Real left   = pos[pId][0] - 2.0f;
            Real front  = 2.0f / 3.0f - pos[pId][2];
            Real behind = pos[pId][2] - 1.0f / 3.0f;
            if (left < behind && left < front)
            {
                pos_i[0] = 2.0f;
            }
            else if (behind < left && behind < front)
            {
                pos_i[2] = 1.0f / 3.0f;
            }
            else
            {
                pos_i[2] = 2.0f / 3.0f;
            }
        }
        else if (pos[pId][2] > 5.0f / 6.0f)
        {
            Real left  = pos[pId][0] - 2.0f;
            Real front = pos[pId][2] - 5.0f / 6.0f;
            if (left < front)
            {
                pos_i[0] = 2.0f;
            }
            else
            {
                pos_i[2] = 5.0f / 6.0f;
            }
        }
        velocities[pId] += (pos_i - pos[pId]) / 0.001f;
        pos[pId] = pos_i;
    }
    if (pos[pId][0] <= 3.0f)
    {
        pos[pId][1] = max(pos[pId][1], 0.0f);
    }
}

template <typename Coord>
__global__ void SFIF_UpdateBoundary_InitFluid(
    DeviceArray<Coord> velocities,
    DeviceArray<Coord> pos)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velocities.size())
        return;

    if (pos[pId][0] >= 1.0f)
    {
        Real delta_pos = pos[pId][0] - 1.0f;
        pos[pId][0]    = 1.0f;
        velocities[pId][0] -= delta_pos / 0.001f;
    }
    if (pos[pId][1] < 0.0f)
        pos[pId][1] = 0.0f;
    // velocities[pId] *= 0.995f;
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

    Function1Pt::copy(velOld, m_vels.getValue());
    for (size_t it = 0; it < 10; it++)
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

    for (size_t it = 0; it < 5; it++)
    {
        weights.reset();
        posBuf.reset();
        K_Collide<<<pDims, BLOCK_SIZE>>>(
            m_objId,
            m_mass.getValue(),
            allpoints,
            posBuf,
            weights,
            m_nbrQuery->outNeighborhood()->getValue(),
            radius.getValue());

        K_ComputeTarget<<<pDims, BLOCK_SIZE>>>(
            allpoints,
            posBuf,
            weights);

        Function1Pt::copy(allpoints, posBuf);
    }

    K_ComputeVelocity<<<pDims, BLOCK_SIZE>>>(init_pos, allpoints, m_vels.getValue(), getParent()->getDt());

    if (frame_count < 500)
    {
        SFIF_UpdateBoundary_InitFluid<<<pDims, BLOCK_SIZE>>>(m_vels.getValue(), allpoints);
    }
    else
    {
        SFIF_UpdateBoundary_Tube<<<pDims, BLOCK_SIZE>>>(m_vels.getValue(), allpoints);
    }
    /*  if (frame_count == 1000)
    {
        output_initialized_particles();
    }*/

    frame_count++;
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