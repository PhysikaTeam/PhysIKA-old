/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Implementation of ParticleViscoplasticBody, simulate viscoplasticity with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleViscoplasticBody.h"

#include "Core/Utility.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/ParticleIntegrator.h"
#include "Dynamics/ParticleSystem/ElastoplasticityModule.h"
#include "Dynamics/ParticleSystem/ImplicitViscosity.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleViscoplasticBody, TDataType)

template <typename TDataType>
ParticleViscoplasticBody<TDataType>::ParticleViscoplasticBody(std::string name)
    : ParticleSystem<TDataType>(name)
{
    m_horizon.setValue(0.0085);

    m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
    this->currentPosition()->connect(m_integrator->inPosition());
    this->currentVelocity()->connect(m_integrator->inVelocity());
    this->currentForce()->connect(m_integrator->inForceDensity());

    m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
    m_horizon.connect(m_nbrQuery->inRadius());
    this->currentPosition()->connect(m_nbrQuery->inPosition());

    m_plasticity = this->template addConstraintModule<ElastoplasticityModule<TDataType>>("elastopolasticity");
    this->currentPosition()->connect(m_plasticity->inPosition());
    this->currentVelocity()->connect(m_plasticity->inVelocity());
    m_nbrQuery->outNeighborhood()->connect(m_plasticity->inNeighborhood());
    m_plasticity->setFrictionAngle(0);
    m_plasticity->setCohesion(0.0);
    m_plasticity->enableFullyReconstruction();

    m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
    m_visModule->setViscosity(Real(1));
    m_horizon.connect(&m_visModule->m_smoothingLength);
    this->currentPosition()->connect(&m_visModule->m_position);
    this->currentVelocity()->connect(&m_visModule->m_velocity);
    m_nbrQuery->outNeighborhood()->connect(&m_visModule->m_neighborhood);

    m_surfaceNode = this->template createChild<Node>("Mesh");
    auto triSet   = std::make_shared<TriangleSet<TDataType>>();
    m_surfaceNode->setTopologyModule(triSet);
    auto render = std::make_shared<SurfaceMeshRender>();
    render->setColor(Vector3f(0.2f, 0.6, 1.0f));
    m_surfaceNode->addVisualModule(render);
    m_surfaceNode->setVisible(false);

    std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
    this->addTopologyMapping(surfaceMapping);
}

template <typename TDataType>
ParticleViscoplasticBody<TDataType>::~ParticleViscoplasticBody()
{
}

template <typename TDataType>
void ParticleViscoplasticBody<TDataType>::advance(Real dt)
{
    m_integrator->begin();

    m_integrator->integrate();

    m_nbrQuery->compute();
    m_plasticity->solveElasticity();
    m_nbrQuery->compute();

    m_plasticity->applyPlasticity();

    m_visModule->constrain();

    m_integrator->end();
}

template <typename TDataType>
void ParticleViscoplasticBody<TDataType>::updateTopology()
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
bool ParticleViscoplasticBody<TDataType>::initialize()
{
    m_nbrQuery->initialize();
    m_nbrQuery->compute();

    return ParticleSystem<TDataType>::initialize();
}

template <typename TDataType>
void ParticleViscoplasticBody<TDataType>::loadSurface(std::string filename)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
}

template <typename TDataType>
bool ParticleViscoplasticBody<TDataType>::translate(Coord t)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

    return ParticleSystem<TDataType>::translate(t);
}

template <typename TDataType>
bool ParticleViscoplasticBody<TDataType>::scale(Real s)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

    return ParticleSystem<TDataType>::scale(s);
}
}  // namespace PhysIKA