/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Implementation of StaticBoundary class, representing static objects in scene that can couple with simulated objects
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#include "StaticBoundary.h"

#include "Framework/Topology/DistanceField3D.h"
#include "Dynamics/ParticleSystem/BoundaryConstraint.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(StaticBoundary, TDataType)

template <typename TDataType>
StaticBoundary<TDataType>::StaticBoundary()
    : Node()
{
}

template <typename TDataType>
StaticBoundary<TDataType>::~StaticBoundary()
{
}

template <typename TDataType>
void StaticBoundary<TDataType>::advance(Real dt)
{
    auto pSys = this->getParticleSystems();

    for (size_t t = 0; t < m_obstacles.size(); t++)
    {
        //coupling with particle systems
        for (int i = 0; i < pSys.size(); i++)
        {
            DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
            DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
            m_obstacles[t]->constrain(posFd->getValue(), velFd->getValue(), dt);
        }
    }
}

template <typename TDataType>
void StaticBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary)
{
    auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
    boundary->load(filename, bOutBoundary);

    m_obstacles.push_back(boundary);
}

template <typename TDataType>
void StaticBoundary<TDataType>::loadCube(Coord lo, Coord hi, Real distance, bool bOutBoundary /*= false*/)
{
    auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
    boundary->setCube(lo, hi, distance, bOutBoundary);
    m_obstacles.push_back(boundary);
}

template <typename TDataType>
void StaticBoundary<TDataType>::loadShpere(Coord center, Real r, Real distance, bool bOutBoundary /*= false*/)
{
    auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
    boundary->setSphere(center, r, distance, bOutBoundary);
    m_obstacles.push_back(boundary);
}

template <typename TDataType>
void StaticBoundary<TDataType>::scale(Real s)
{
    for (int i = 0; i < m_obstacles.size(); i++)
    {
        m_obstacles[i]->m_cSDF->scale(s);
    }
}

template <typename TDataType>
void StaticBoundary<TDataType>::translate(Coord t)
{
    for (int i = 0; i < m_obstacles.size(); i++)
    {
        m_obstacles[i]->m_cSDF->translate(t);
    }
}
}  // namespace PhysIKA