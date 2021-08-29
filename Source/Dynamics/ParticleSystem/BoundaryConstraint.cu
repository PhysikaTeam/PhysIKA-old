/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Implementation of BoundaryConstraint class, SDF based collision handling constraint
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-26
 * @description: poslish code
 * @version    : 1.1
 */

#include "BoundaryConstraint.h"

#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Framework/Topology/DistanceField3D.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(BoundaryConstraint, TDataType)

template <typename TDataType>
BoundaryConstraint<TDataType>::BoundaryConstraint()
    : ConstraintModule()
{
    Coord lo(0.0f);
    Coord hi(1.0f);
    m_cSDF = std::make_shared<DistanceField3D<DataType3f>>();
    m_cSDF->setSpace(lo - 0.025f, hi + 0.025f, 105, 105, 105);
    m_cSDF->loadBox(lo, hi, true);
}

template <typename TDataType>
BoundaryConstraint<TDataType>::~BoundaryConstraint()
{
    m_cSDF->release();
}

/**
 * handle collision between specified DOF and SDF
 * @param[in&&out] posArr               positions of the simulated objects
 * @param[in&&out] velArr               velocities of the simulated objects
 * @param[in]      df                   sdf of the static boundary
 * @param[in]      normalFriction       friction coefficient in normal direction
 * @param[in]      tangentialFriction   friction coeffciient in tangential direction
 * @param[in]      dt                   time step
 */
template <typename Real, typename Coord, typename TDataType>
__global__ void K_ConstrainSDF(
    DeviceArray<Coord>         posArr,
    DeviceArray<Coord>         velArr,
    DistanceField3D<TDataType> df,
    Real                       normalFriction,
    Real                       tangentialFriction,
    Real                       dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size())
        return;

    Coord pos = posArr[pId];
    Coord vec = velArr[pId];

    Real  dist;
    Coord normal;
    df.getDistance(pos, dist, normal);
    // constrain particle
    if (dist <= 0)
    {
        Real       olddist = -dist;
        RandNumber rGen(pId);
        dist = 0.0001f * rGen.Generate();
        // reflect position
        pos -= (olddist + dist) * normal;
        // reflect velocity
        Real  vlength    = vec.norm();
        Real  vec_n      = vec.dot(normal);
        Coord vec_normal = vec_n * normal;
        Coord vec_tan    = vec - vec_normal;
        if (vec_n > 0)
            vec_normal = -vec_normal;
        vec_normal *= (1.0f - normalFriction);  //linear frictional model in normal direction
        vec = vec_normal + vec_tan;
        vec *= pow(Real(M_E), -dt * tangentialFriction);  //exponential friction model in tangential direction
    }

    posArr[pId] = pos;
    velArr[pId] = vec;
}

template <typename TDataType>
bool BoundaryConstraint<TDataType>::constrain()
{
    cuint pDim = cudaGridSize(m_position.getElementCount(), BLOCK_SIZE);
    K_ConstrainSDF<<<pDim, BLOCK_SIZE>>>(
        m_position.getValue(),
        m_velocity.getValue(),
        *m_cSDF,
        m_normal_friction,
        m_tangent_friction,
        getParent()->getDt());

    return true;
}

template <typename TDataType>
bool BoundaryConstraint<TDataType>::constrain(DeviceArray<Coord>& position, DeviceArray<Coord>& velocity, Real dt)
{
    cuint pDim = cudaGridSize(position.size(), BLOCK_SIZE);
    K_ConstrainSDF<<<pDim, BLOCK_SIZE>>>(
        position,
        velocity,
        *m_cSDF,
        m_normal_friction,
        m_tangent_friction,
        dt);

    return true;
}

template <typename TDataType>
void BoundaryConstraint<TDataType>::load(std::string filename, bool inverted)
{
    m_cSDF->loadSDF(filename, inverted);
}

template <typename TDataType>
void BoundaryConstraint<TDataType>::setCube(Coord lo, Coord hi, Real distance, bool inverted)
{
    int nx = floorf((hi[0] - lo[0]) / distance);
    int ny = floorf((hi[1] - lo[1]) / distance);
    int nz = floorf((hi[2] - lo[2]) / distance);

    m_cSDF->setSpace(lo - 5 * distance, hi + 5 * distance, nx + 10, ny + 10, nz + 10);
    m_cSDF->loadBox(lo, hi, inverted);
}

template <typename TDataType>
void BoundaryConstraint<TDataType>::setSphere(Coord center, Real r, Real distance, bool inverted)
{
    int nx = floorf(2 * r / distance);

    m_cSDF->setSpace(center - r - 5 * distance, center + r + 5 * distance, nx + 10, nx + 10, nx + 10);
    m_cSDF->loadSphere(center, r, inverted);
}

}  // namespace PhysIKA