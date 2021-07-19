/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Implementation of ParticleCloth class, a scene node to simulate cloth with peridynamics
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-14
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleCloth.h"

#include "Core/Utility.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/Peridynamics.h"
#include "Rendering/SurfaceMeshRender.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleCloth, TDataType)

template <typename TDataType>
ParticleCloth<TDataType>::ParticleCloth(std::string name)
    : ParticleSystem<TDataType>(name)
{
    //set numerical model
    m_peri = std::make_shared<Peridynamics<TDataType>>();
    this->setNumericalModel(m_peri);
    this->currentPosition()->connect(&m_peri->m_position);
    this->currentVelocity()->connect(&m_peri->m_velocity);
    this->currentForce()->connect(&m_peri->m_forceDensity);

    //Create a node for surface mesh rendering
    m_surfaceNode = this->template createChild<Node>("Mesh");
    auto render   = std::make_shared<SurfaceMeshRender>();
    render->setColor(Vector3f(0.4, 0.75, 1));
    m_surfaceNode->addVisualModule(render);

    //topology mapping between simulation particles and surface mesh
    auto triSet = std::make_shared<TriangleSet<TDataType>>();
    m_surfaceNode->setTopologyModule(triSet);
    std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
    this->addTopologyMapping(surfaceMapping);
    // this->setVisible(false);
}

template <typename TDataType>
ParticleCloth<TDataType>::~ParticleCloth()
{
}

template <typename TDataType>
bool ParticleCloth<TDataType>::translate(Coord t)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

    return ParticleSystem<TDataType>::translate(t);
}

template <typename TDataType>
bool ParticleCloth<TDataType>::scale(Real s)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

    return ParticleSystem<TDataType>::scale(s);
}

template <typename TDataType>
bool ParticleCloth<TDataType>::initialize()
{
    return ParticleSystem<TDataType>::initialize();
}

template <typename TDataType>
void ParticleCloth<TDataType>::advance(Real dt)
{
    this->setDt(dt);
    m_peri->m_elasticity->setIterationNumber(3);
    m_peri->step(this->getDt());
}

template <typename TDataType>
void ParticleCloth<TDataType>::updateTopology()
{
    auto pts = this->m_pSet->getPoints();
    Function1Pt::copy(pts, this->currentPosition()->getValue());

    auto tMappings = this->getTopologyMappingList();
    for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
    {
        (*iter)->apply();
    }
}

template <typename TDataType>
void ParticleCloth<TDataType>::loadSurface(std::string filename)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
}
}  // namespace PhysIKA