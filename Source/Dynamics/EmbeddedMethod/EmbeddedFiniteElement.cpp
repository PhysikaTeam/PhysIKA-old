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

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(EmbeddedFiniteElement, TDataType)

	template<typename TDataType>
	EmbeddedFiniteElement<TDataType>::EmbeddedFiniteElement(std::string name)
  : ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);
		this->attachField(&m_horizon, "horizon", "horizon");

    auto m_integrator = this->template setNumericalIntegrator<EmbeddedIntegrator<TDataType>>("integrator");
    this->getPosition()->connect(m_integrator->m_position);
		this->getVelocity()->connect(m_integrator->m_velocity);
		this->getForce()->connect(m_integrator->m_forceDensity);

		auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->getPosition()->connect(m_elasticity->m_position);
		this->getVelocity()->connect(m_elasticity->m_velocity);
		m_horizon.connect(m_elasticity->m_horizon);
		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

		//Set the topology mapping from PointSet to TriangleSet
		auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
		surfaceMapping->setFrom(this->m_pSet);
		surfaceMapping->setTo(triSet);
	}

	template<typename TDataType>
	EmbeddedFiniteElement<TDataType>::~EmbeddedFiniteElement()
	{

	}

	template<typename TDataType>
	bool EmbeddedFiniteElement<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool EmbeddedFiniteElement<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool EmbeddedFiniteElement<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void EmbeddedFiniteElement<TDataType>::advance(Real dt)
	{
		auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		integrator->begin();

		integrator->integrate();

    // if (module != nullptr)
    //   module->constrain();

		integrator->end();
	}

	template<typename TDataType>
	void EmbeddedFiniteElement<TDataType>::updateTopology()
	{

		auto pts = this->m_pSet->getPoints();
		Function1Pt::copy(pts, this->getPosition()->getValue());

    /*TODO:fix bug:
      apply() will not update points in triSet because points in triSet has no neighbours */
		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}

    //So we just copy the points. WARNING: the surface should have the same points as pointsSets.
    auto triSet = m_surfaceNode->template getModule<TriangleSet<TDataType>>("surface_mesh");
    Function1Pt::copy(triSet->getPoints(), pts);
	}


	template<typename TDataType>
	std::shared_ptr<ElasticityModule<TDataType>> EmbeddedFiniteElement<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void EmbeddedFiniteElement<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
	{
		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		this->getPosition()->connect(solver->m_position);
		this->getVelocity()->connect(solver->m_velocity);
		nbrQuery->m_neighborhood.connect(solver->m_neighborhood);
		m_horizon.connect(solver->m_horizon);

		this->deleteModule(module);

		solver->setName("elasticity");
		this->addConstraintModule(solver);
	}


	template<typename TDataType>
	void EmbeddedFiniteElement<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}


	template<typename TDataType>
	std::shared_ptr<PointSetToPointSet<TDataType>> EmbeddedFiniteElement<TDataType>::getTopologyMapping()
	{
		auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

		return mapping;
	}

  template<typename TDataType>
  void EmbeddedFiniteElement<TDataType>::init_problem_and_solver(const boost::property_tree::ptree& pt)
  {
    auto& m_coords = ParticleSystem<TDataType>::m_pSet->getPoints();
    HostArray<Coord> pts(m_coords.size());
    Function1Pt::copy(pts, m_coords);
    const size_t num = pts.size();
    std::vector<Real> nods(3 * num);
#pragma omp parallel for
    for(size_t i = 0; i < num; ++i)
      for(size_t j = 0; j < 3; ++j)
        nods[j + 3 * i] = pts[i][j];

    epb_fac  = std::make_shared<embedded_elas_problem_builder<Real>>(&nods[0], pt);

    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    integrator->bind_problem(epb_fac, pt);

  }
}
