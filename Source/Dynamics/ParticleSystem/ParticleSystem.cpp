/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Implementation of ParticleSystem class, base class of all particle-based methods
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-20
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleSystem.h"

#include "Core/Utility.h"
#include "Framework/Topology/PointSet.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleSystem, TDataType)

template <typename TDataType>
ParticleSystem<TDataType>::ParticleSystem(std::string name)
    : Node(name)
{
    m_pSet = std::make_shared<PointSet<TDataType>>();
    this->setTopologyModule(m_pSet);
}

template <typename TDataType>
ParticleSystem<TDataType>::~ParticleSystem()
{
}

template <typename TDataType>
void ParticleSystem<TDataType>::loadParticles(std::string filename)
{
    m_pSet->loadObjFile(filename);
    printf("ParticleNum:  %d\n", m_pSet->getPointSize());
}

template <typename TDataType>
void ParticleSystem<TDataType>::loadParticles(Coord center, Real r, Real distance)
{
    std::vector<Coord> vertList;
    std::vector<Coord> normalList;

    Coord lo = center - r;
    Coord hi = center + r;

    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real y = lo[1]; y <= hi[1]; y += distance)
        {
            for (Real z = lo[2]; z <= hi[2]; z += distance)
            {
                Coord p = Coord(x, y, z);
                if ((p - center).norm() < r)
                {
                    vertList.push_back(Coord(x, y, z));
                }
            }
        }
    }
    normalList.resize(vertList.size());

    m_pSet->setPoints(vertList);
    m_pSet->setNormals(normalList);

    vertList.clear();
    normalList.clear();
}

template <typename TDataType>
void ParticleSystem<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
{
    std::vector<Coord> vertList;
    std::vector<Coord> normalList;

    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real y = lo[1]; y <= hi[1]; y += distance)
        {
            for (Real z = lo[2]; z <= hi[2]; z += distance)
            {
                Coord p = Coord(x, y, z);
                vertList.push_back(Coord(x, y, z));
            }
        }
    }
    normalList.resize(vertList.size());

    m_pSet->setPoints(vertList);
    m_pSet->setNormals(normalList);

    std::cout << "particle number: " << vertList.size() << std::endl;

    vertList.clear();
    normalList.clear();
}

template <typename TDataType>
bool ParticleSystem<TDataType>::translate(Coord t)
{
    m_pSet->translate(t);

    return true;
}

template <typename TDataType>
bool ParticleSystem<TDataType>::scale(Real s)
{
    m_pSet->scale(s);

    return true;
}

template <typename TDataType>
bool ParticleSystem<TDataType>::initialize()
{
    return Node::initialize();
}

template <typename TDataType>
void ParticleSystem<TDataType>::updateTopology()
{
    if (!this->currentPosition()->isEmpty())
    {
        int   num = this->currentPosition()->getElementCount();
        auto& pts = m_pSet->getPoints();
        if (num != pts.size())
        {
            pts.resize(num);
        }

        Function1Pt::copy(pts, this->currentPosition()->getValue());
    }
}

template <typename TDataType>
bool ParticleSystem<TDataType>::resetStatus()
{
    auto pts = m_pSet->getPoints();

    if (pts.size() > 0)
    {
        this->currentPosition()->setElementCount(pts.size());
        this->currentVelocity()->setElementCount(pts.size());
        this->currentForce()->setElementCount(pts.size());

        Function1Pt::copy(this->currentPosition()->getValue(), pts);
        this->currentVelocity()->getReference()->reset();
    }

    return Node::resetStatus();
}

}  // namespace PhysIKA