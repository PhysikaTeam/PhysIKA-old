/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2021-06-17
 * @description: Implementation of ParticleFluidFast class
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-21
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleFluidFast.h"

#include <thrust/sort.h>

#include "Core/Utility.h"
#include "Framework/Topology/PointSet.h"
#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleFluidFast, TDataType)

template <typename TDataType>
ParticleFluidFast<TDataType>::ParticleFluidFast(std::string name)
    : ParticleSystem<TDataType>(name)
{
    auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
    this->setNumericalModel(pbf);

    this->currentPositionInOrder()->connect(&pbf->m_position);
    this->currentVelocityInOrder()->connect(&pbf->m_velocity);
    this->currentForceInOrder()->connect(&pbf->m_forceDensity);
}

template <typename TDataType>
ParticleFluidFast<TDataType>::~ParticleFluidFast()
{
}

/**
 * calcute particle's ids and corresponding hash grid ids according to particle positions
 *
 * @param[out] ids           grid id that the particle is in
 * @param[out] idsInOrder    thread id of each particle
 * @param[in]  poss          positions of the particles
 */
template <typename Coord>
__global__ void CalculateIds(DeviceArray<int> ids, DeviceArray<int> idsInOrder, DeviceArray<Coord> poss)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= poss.size())
        return;

    Coord lo = Coord(-0.5, 0, -0.5);
    Coord hi = Coord(1.5, 1, 1.5);

    Real spacing = Real(0.01);

    int nx_max = (hi[0] - lo[0]) / spacing;
    int ny_max = (hi[1] - lo[1]) / spacing;
    int nz_max = (hi[2] - lo[2]) / spacing;

    Coord p  = poss[pId];
    int   nx = (p[0] - lo[0]) / spacing;
    int   ny = (p[1] - lo[1]) / spacing;
    int   nz = (p[2] - lo[2]) / spacing;

    nx = clamp(nx, 0, nx_max);
    ny = clamp(ny, 0, ny_max);
    nz = clamp(nz, 0, nz_max);

    ids[pId]        = nx + ny * nx_max + nz * nx_max * ny_max;
    idsInOrder[pId] = pId;
}

/**
 * reorder array content to be consistent with the ids
 *
 * @param[out] newArray    the array to store reordered content
 * @param[in]  oldArray    the array before reorder
 * @param[in]  ids         the ids used to reorder the input array
 */
template <typename Coord>
__global__ void ReorderArray(DeviceArray<Coord> newArray, DeviceArray<Coord> oldArray, DeviceArray<int> ids)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= ids.size())
        return;

    newArray[pId] = oldArray[ids[pId]];
}

template <typename TDataType>
void ParticleFluidFast<TDataType>::advance(Real dt)
{
    int total_num = this->currentPosition()->getElementCount();

    if (ids.size() != total_num)
    {
        ids.resize(total_num);
    }

    if (idsInOrder.size() != total_num)
    {
        idsInOrder.resize(total_num);
    }

    cuExecute(ids.size(),
              CalculateIds,
              ids,
              idsInOrder,
              this->currentPosition()->getValue());

    // sort ids and idsInOrder simultaneouly in ascending order of key
    thrust::sort_by_key(thrust::device, ids.begin(), ids.begin() + ids.size(), idsInOrder.begin());

    if (this->currentPositionInOrder()->getElementCount() != total_num)
    {
        this->currentPositionInOrder()->setElementCount(total_num);
        this->currentVelocityInOrder()->setElementCount(total_num);
        this->currentForceInOrder()->setElementCount(total_num);
    }

    //reorder particle states
    cuExecute(total_num,
              ReorderArray,
              this->currentPositionInOrder()->getValue(),
              this->currentPosition()->getValue(),
              idsInOrder);

    cuExecute(total_num,
              ReorderArray,
              this->currentVelocityInOrder()->getValue(),
              this->currentVelocity()->getValue(),
              idsInOrder);

    cuExecute(total_num,
              ReorderArray,
              this->currentForceInOrder()->getValue(),
              this->currentForce()->getValue(),
              idsInOrder);

    //apply simulation
    if (total_num > 0 && this->self_update)
    {
        auto nModel = this->getNumericalModel();
        nModel->step(this->getDt());
    }

    this->currentPosition()->setValue(this->currentPositionInOrder()->getValue());
    this->currentVelocity()->setValue(this->currentVelocityInOrder()->getValue());
    this->currentForce()->setValue(this->currentForceInOrder()->getValue());
}

template <typename TDataType>
bool ParticleFluidFast<TDataType>::resetStatus()
{
    std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();
    if (m_particleEmitters.size() > 0)
    {
        this->currentPosition()->setElementCount(0);
        this->currentVelocity()->setElementCount(0);
        this->currentForce()->setElementCount(0);
    }
    else
        return ParticleSystem<TDataType>::resetStatus();
    return true;
}

}  // namespace PhysIKA