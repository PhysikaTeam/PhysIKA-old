#include "EmbeddedFiniteElement.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "EmbeddedIntegrator.h"
#include "Problem/integrated_problem/embedded_elas_fem_problem.h"
#include "Solver/newton_method.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include <iostream>

using namespace std;

namespace PhysIKA {
IMPLEMENT_CLASS_1(EmbeddedFiniteElement, TDataType)

template <typename TDataType>
EmbeddedFiniteElement<TDataType>::EmbeddedFiniteElement(std::string name)
    : ParticleSystem<TDataType>(name)
{
    //m_horizon.setValue(0.0085);
    this->varHorizon()->setValue(0.0085);
    //this->attachField(&m_horizon, "horizon", "horizon");

    auto m_integrator = this->template setNumericalIntegrator<EmbeddedIntegrator<TDataType>>("integrator");
    this->currentPosition()->connect(m_integrator->inPosition());
    this->currentVelocity()->connect(m_integrator->inVelocity());
    this->currentForce()->connect(m_integrator->inForceDensity());

    /*this->getPosition()->connect2(m_integrator->m_position);
        this->getVelocity()->connect2(m_integrator->m_velocity);
        this->getForce()->connect2(m_integrator->m_forceDensity);*/

    this->getAnimationPipeline()->push_back(m_integrator);

    auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
    /*m_horizon.connect2(m_nbrQuery->in_Radius);
        this->current_Position.connect2(m_nbrQuery->in_Position);*/
    this->varHorizon()->connect(m_nbrQuery->inRadius());
    this->currentPosition()->connect(m_nbrQuery->inPosition());

    this->getAnimationPipeline()->push_back(m_nbrQuery);

    auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
    this->varHorizon()->connect(m_elasticity->inHorizon());
    this->currentPosition()->connect(m_elasticity->inPosition());
    this->currentVelocity()->connect(m_elasticity->inVelocity());
    m_nbrQuery->outNeighborhood()->connect(m_elasticity->inNeighborhood());

    /*    this->getPosition()->connect2(m_elasticity->in_Position);
        this->getVelocity()->connect2(m_elasticity->in_Velocity);
        m_horizon.connect2(m_elasticity->in_Horizon);
        m_nbrQuery->out_Neighborhood.connect2(m_elasticity->in_Neighborhood);*/

    this->getAnimationPipeline()->push_back(m_elasticity);

    //Create a node for surface mesh rendering
    m_surfaceNode = this->template createChild<Node>("Mesh");

    auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

    //Set the topology mapping from PointSet to TriangleSet
    auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
    surfaceMapping->setFrom(this->m_pSet);
    surfaceMapping->setTo(triSet);
}

template <typename TDataType>
EmbeddedFiniteElement<TDataType>::~EmbeddedFiniteElement()
{
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::translate(Coord t)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

    return ParticleSystem<TDataType>::translate(t);
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::scale(Real s)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

    return ParticleSystem<TDataType>::scale(s);
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::initialize()
{
    return ParticleSystem<TDataType>::initialize();
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::advance(Real dt)
{
    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    auto module     = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    integrator->begin();

    integrator->integrate();

    /*     if (module != nullptr)
           module->constrain();*/

    integrator->end();
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::updateTopology()
{

    auto pts = this->m_pSet->getPoints();
    Function1Pt::copy(pts, this->currentPosition()->getValue());

    /*TODO:fix bug:
          apply() will not update points in triSet because points in triSet has no neighbours */
    auto tMappings = this->getTopologyMappingList();
    for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
    {
        (*iter)->apply();
    }

    //So we just copy the points. WARNING: the surface should have the same points as pointsSets.
    // auto triSet = m_surfaceNode->template getModule<TriangleSet<TDataType>>("surface_mesh");
    // Function1Pt::copy(triSet->getPoints(), pts);
}

template <typename TDataType>
std::shared_ptr<ElasticityModule<TDataType>> EmbeddedFiniteElement<TDataType>::getElasticitySolver()
{
    auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
    return module;
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
{
    auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
    auto module   = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    /*    this->getPosition()->connect2(solver->in_Position);
        this->getVelocity()->connect2(solver->in_Velocity);
        nbrQuery->out_Neighborhood.connect2(solver->in_Neighborhood);
        m_horizon.connect2(solver->in_Horizon);*/

    this->currentPosition()->connect(solver->inPosition());
    this->currentVelocity()->connect(solver->inVelocity());
    nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
    this->varHorizon()->connect(solver->inHorizon());

    this->deleteModule(module);

    solver->setName("elasticity");
    this->addConstraintModule(solver);
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::loadSurface(std::string filename)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
}

template <typename TDataType>
std::shared_ptr<PointSetToPointSet<TDataType>> EmbeddedFiniteElement<TDataType>::getTopologyMapping()
{
    auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

    return mapping;
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::init_problem_and_solver(const boost::property_tree::ptree& pt)
{
    auto&            m_coords = ParticleSystem<TDataType>::m_pSet->getPoints();
    HostArray<Coord> pts(m_coords.size());
    Function1Pt::copy(pts, m_coords);
    const size_t      num = pts.size();
    std::vector<Real> nods(3 * num);
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i)
        for (size_t j = 0; j < 3; ++j)
            nods[j + 3 * i] = pts[i][j];

    epb_fac = std::make_shared<embedded_elas_problem_builder<Real>>(&nods[0], pt);

    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    integrator->bind_problem(epb_fac, pt);
}
}  // namespace PhysIKA