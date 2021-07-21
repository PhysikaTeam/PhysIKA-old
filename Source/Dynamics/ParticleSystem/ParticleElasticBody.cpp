/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Implementation of ParticleElasticBody class, projective-peridynamics based elastic bodies
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-20
 * @description: poslish code
 * @version    : 1.1
 */
#include "ParticleElasticBody.h"

#include "Core/Utility.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElasticityModule.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

template <typename TDataType>
ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
    : ParticleSystem<TDataType>(name)
{
    this->varHorizon()->setValue(0.0085);

    /*note on connect operation of VarField:
     it leads to memory sharing between 2 fields,
     A->connect(B) means B uses A's memory
     */

    //register integrator
    auto m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
    this->currentPosition()->connect(m_integrator->inPosition());
    this->currentVelocity()->connect(m_integrator->inVelocity());
    this->currentForce()->connect(m_integrator->inForceDensity());
    this->getAnimationPipeline()->push_back(m_integrator);

    //register neighbor query operation
    auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
    this->varHorizon()->connect(m_nbrQuery->inRadius());
    this->currentPosition()->connect(m_nbrQuery->inPosition());
    this->getAnimationPipeline()->push_back(m_nbrQuery);

    //register elasticity module
    auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
    this->varHorizon()->connect(m_elasticity->inHorizon());
    this->currentPosition()->connect(m_elasticity->inPosition());
    this->currentVelocity()->connect(m_elasticity->inVelocity());
    m_nbrQuery->outNeighborhood()->connect(m_elasticity->inNeighborhood());
    this->getAnimationPipeline()->push_back(m_elasticity);

    //Create a node for surface mesh rendering
    m_surfaceNode = this->template createChild<Node>("Mesh");
    auto triSet   = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

    //Set the topology mapping from PointSet to TriangleSet
    //the mapping will be used to deform the mesh according to particle configuration
    auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
    surfaceMapping->setFrom(this->m_pSet);  //source
    surfaceMapping->setTo(triSet);          //target
}

template <typename TDataType>
ParticleElasticBody<TDataType>::~ParticleElasticBody()
{
}

template <typename TDataType>
bool ParticleElasticBody<TDataType>::translate(Coord t)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

    return ParticleSystem<TDataType>::translate(t);
}

template <typename TDataType>
bool ParticleElasticBody<TDataType>::scale(Real s)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

    return ParticleSystem<TDataType>::scale(s);
}

template <typename TDataType>
bool ParticleElasticBody<TDataType>::initialize()
{
    return ParticleSystem<TDataType>::initialize();
}

template <typename TDataType>
void ParticleElasticBody<TDataType>::advance(Real dt)
{
    auto integrator = this->template getModule<ParticleIntegrator<TDataType>>("integrator");
    auto module     = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    integrator->begin();
    integrator->integrate();
    if (module != nullptr)
        module->constrain();
    integrator->end();
}

template <typename TDataType>
void ParticleElasticBody<TDataType>::updateTopology()
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
std::shared_ptr<ElasticityModule<TDataType>> ParticleElasticBody<TDataType>::getElasticitySolver()
{
    auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
    return module;
}

template <typename TDataType>
void ParticleElasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
{
    auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
    auto module   = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    this->currentPosition()->connect(solver->inPosition());
    this->currentVelocity()->connect(solver->inVelocity());
    nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
    this->varHorizon()->connect(solver->inHorizon());

    this->deleteModule(module);

    solver->setName("elasticity");
    this->addConstraintModule(solver);
}

template <typename TDataType>
void ParticleElasticBody<TDataType>::loadSurface(std::string filename)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
}

template <typename TDataType>
std::shared_ptr<PointSetToPointSet<TDataType>> ParticleElasticBody<TDataType>::getTopologyMapping()
{
    auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

    return mapping;
}

}  // namespace PhysIKA