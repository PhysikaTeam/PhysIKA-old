#include "ParticleElastoplasticBody.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/TriangleSet.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/SurfaceMeshRender.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Peridynamics.h"
#include "Physika_Framework/Mapping/PointSetToPointSet.h"
#include "Physika_Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElastoplasticityModule.h"

#include "DensityPBD.h"
#include "ImplicitViscosity.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleElastoplasticBody, TDataType)

	template<typename TDataType>
	ParticleElastoplasticBody<TDataType>::ParticleElastoplasticBody(std::string name)
		: ParticleSystem(name)
	{
		m_horizon.setValue(0.0085);

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		m_position.connect(m_integrator->m_position);
		m_velocity.connect(m_integrator->m_velocity);
		m_force.connect(m_integrator->m_forceDensity);
		//m_integrator->initialize();
		

		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_horizon.connect(m_nbrQuery->m_radius);
		m_position.connect(m_nbrQuery->m_position);

		//m_nbrQuery->initialize();
		//m_nbrQuery->compute();

		m_plasticity = std::make_shared<ElastoplasticityModule<TDataType>>();
		m_position.connect(m_plasticity->m_position);
		m_velocity.connect(m_plasticity->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_plasticity->m_neighborhood);
		//m_plasticity->initialize();

		m_pbdModule = std::make_shared<DensityPBD<TDataType>>();
		m_horizon.connect(m_pbdModule->m_smoothingLength);
		m_position.connect(m_pbdModule->m_position);
		m_velocity.connect(m_pbdModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_pbdModule->m_neighborhood);
		//m_pbdModule->initialize();

		m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->setViscosity(Real(1));
		m_horizon.connect(m_visModule->m_smoothingLength);
		m_position.connect(m_visModule->m_position);
		m_velocity.connect(m_visModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_visModule->m_neighborhood);
		//m_visModule->initialize();

		this->addModule(m_integrator);
		this->addModule(m_nbrQuery);
		this->addConstraintModule(m_plasticity);
		this->addConstraintModule(m_pbdModule);
		this->addConstraintModule(m_visModule);
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
//		m_pbdModule->constrain();
//		m_visModule->constrain();
//		m_elasticity->constrain();
//		m_plasticity->constrain();
		m_plasticity->solveElasticity();
		m_nbrQuery->compute();

		m_plasticity->solvePlasticity();

		m_visModule->constrain();

		m_integrator->end();
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::updateTopology()
	{
		auto pts = m_pSet->getPoints();
		Function1Pt::copy(pts, getPosition()->getValue());

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

}