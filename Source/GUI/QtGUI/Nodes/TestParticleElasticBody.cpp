#include "TestParticleElasticBody.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/ParticleIntegrator.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(TestParticleElasticBody, TDataType)

	template<typename TDataType>
	TestParticleElasticBody<TDataType>::TestParticleElasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);
		this->attachField(&m_horizon, "horizon", "horizon");
		this->attachField(&m_test, "test", "This is a test for vector3");

		auto m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->getPosition()->connect(m_integrator->m_position);
		this->getVelocity()->connect(m_integrator->m_velocity);
		this->getForce()->connect(m_integrator->m_forceDensity);

		this->getAnimationPipeline()->push_back(m_integrator);

		auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		this->getAnimationPipeline()->push_back(m_nbrQuery);


		auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->getPosition()->connect(m_elasticity->m_position);
		this->getVelocity()->connect(m_elasticity->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);

		m_nbrQuery->getTestOut().connect(m_elasticity->getTestIn());

		this->getAnimationPipeline()->push_back(m_elasticity);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

		//Set the topology mapping from PointSet to TriangleSet
		auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
		surfaceMapping->setFrom(this->m_pSet);
		surfaceMapping->setTo(triSet);
	}

	template<typename TDataType>
	TestParticleElasticBody<TDataType>::~TestParticleElasticBody()
	{
		
	}

	template<typename TDataType>
	bool TestParticleElasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool TestParticleElasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool TestParticleElasticBody<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void TestParticleElasticBody<TDataType>::advance(Real dt)
	{
		auto integrator = this->template getModule<ParticleIntegrator<TDataType>>("integrator");

		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		integrator->begin();

		integrator->integrate();

		if (module != nullptr)
			module->constrain();

		integrator->end();
	}

	template<typename TDataType>
	void TestParticleElasticBody<TDataType>::updateTopology()
	{
		auto pts = this->m_pSet->getPoints();
		Function1Pt::copy(pts, this->getPosition()->getValue());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}


	template<typename TDataType>
	std::shared_ptr<ElasticityModule<TDataType>> TestParticleElasticBody<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void TestParticleElasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
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
	void TestParticleElasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}


	template<typename TDataType>
	std::shared_ptr<PointSetToPointSet<TDataType>> TestParticleElasticBody<TDataType>::getTopologyMapping()
	{
		auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

		return mapping;
	}

}