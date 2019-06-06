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

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->m_position.connect(m_integrator->m_position);
		this->m_velocity.connect(m_integrator->m_velocity);
		this->m_force.connect(m_integrator->m_forceDensity);
		//m_integrator->initialize();
		
		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		//m_nbrQuery->initialize();
		//m_nbrQuery->compute();

		m_plasticity = std::make_shared<ElastoplasticityModule<TDataType>>();
		this->m_position.connect(m_plasticity->m_position);
		this->m_velocity.connect(m_plasticity->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_plasticity->m_neighborhood);
		//m_plasticity->initialize();

		m_pbdModule = std::make_shared<DensityPBD<TDataType>>();
		m_horizon.connect(m_pbdModule->m_smoothingLength);
		this->m_position.connect(m_pbdModule->m_position);
		this->m_velocity.connect(m_pbdModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_pbdModule->m_neighborhood);
		//m_pbdModule->initialize();

		m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->setViscosity(Real(1));
		m_horizon.connect(m_visModule->m_smoothingLength);
		this->m_position.connect(m_visModule->m_position);
		this->m_velocity.connect(m_visModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_visModule->m_neighborhood);
		//m_visModule->initialize();

		this->addModule(m_integrator);
		this->addModule(m_nbrQuery);
		this->addConstraintModule(m_plasticity);
		this->addConstraintModule(m_pbdModule);
		this->addConstraintModule(m_visModule);


		m_surfaceNode = this->template createChild<Node>("Mesh");

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
		m_integrator->begin();

		m_integrator->integrate();

		m_nbrQuery->compute();
		m_plasticity->solveElasticity();
		m_nbrQuery->compute();

		m_plasticity->applyPlasticity();

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
}