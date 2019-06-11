#include "ParticleElastoplasticBody.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "Core/Utility.h"
#include "Peridynamics.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElastoplasticityModule.h"

#include "DensityPBD.h"
#include "ImplicitViscosity.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleElastoplasticBody, TDataType)

	template<typename TDataType>
	ParticleElastoplasticBody<TDataType>::ParticleElastoplasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->m_position.connect(m_integrator->m_position);
		this->m_velocity.connect(m_integrator->m_velocity);
		this->m_force.connect(m_integrator->m_forceDensity);
		
		m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		m_plasticity = this->template addConstraintModule<ElastoplasticityModule<TDataType>>("elastoplasticity");
		this->m_position.connect(m_plasticity->m_position);
		this->m_velocity.connect(m_plasticity->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_plasticity->m_neighborhood);

		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("pbd");
		m_horizon.connect(m_pbdModule->m_smoothingLength);
		this->m_position.connect(m_pbdModule->m_position);
		this->m_velocity.connect(m_pbdModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_pbdModule->m_neighborhood);

		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		m_horizon.connect(m_visModule->m_smoothingLength);
		this->m_position.connect(m_visModule->m_position);
		this->m_velocity.connect(m_visModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_visModule->m_neighborhood);


		m_surfaceNode = this->template createChild<Node>("Mesh");
		m_surfaceNode->setVisible(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vector3f(0.2f, 0.6, 1.0f));
		m_surfaceNode->addVisualModule(render);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	ParticleElastoplasticBody<TDataType>::~ParticleElastoplasticBody()
	{
		
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::advance(Real dt)
	{
		auto module = this->template getModule<ElastoplasticityModule<TDataType>>("elastoplasticity");

		m_integrator->begin();

		m_integrator->integrate();

		m_nbrQuery->compute();
		module->solveElasticity();
		m_nbrQuery->compute();

		module->applyPlasticity();

		m_visModule->constrain();

		m_integrator->end();
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::updateTopology()
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
	bool ParticleElastoplasticBody<TDataType>::initialize()
	{
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	bool ParticleElastoplasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ParticleElastoplasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver)
	{
		auto module = this->getModule("elastoplasticity");
		this->deleteModule(module);

		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");

		this->getPosition()->connect(solver->m_position);
		this->getVelocity()->connect(solver->m_velocity);
		nbrQuery->m_neighborhood.connect(solver->m_neighborhood);
		m_horizon.connect(solver->m_horizon);

		solver->setName("elastoplasticity");
		this->addConstraintModule(solver);
	}
}