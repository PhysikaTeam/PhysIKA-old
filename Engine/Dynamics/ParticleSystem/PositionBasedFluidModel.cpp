#include "PositionBasedFluidModel.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticleIntegrator.h"
#include "DensitySummation.h"
#include "ImplicitViscosity.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/Helmholtz.h"
#include "Dynamics/ParticleSystem/Attribute.h"
#include "Core/Utility.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(PositionBasedFluidModel, TDataType)

	template<typename TDataType>
	PositionBasedFluidModel<TDataType>::PositionBasedFluidModel()
		: NumericalModel()
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
		m_smoothingLength.setValue(Real(0.0075));

		attachField(&m_smoothingLength, "smoothingLength", "Smoothing length", false);

		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_forceDensity, "force_density", "Storing the particle force densities!", false);
	}

	template<typename TDataType>
	PositionBasedFluidModel<TDataType>::~PositionBasedFluidModel()
	{
		
	}

	template<typename TDataType>
	bool PositionBasedFluidModel<TDataType>::initializeImpl()
	{
		this->NumericalModel::initializeImpl();

		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		PointSet<TDataType>* pSet = dynamic_cast<PointSet<TDataType>*>(parent->getTopologyModule().get());
		if (pSet == NULL)
		{
			Log::sendMessage(Log::Error, "The topology module is not supported!");
			return false;
		}

		if (!pSet->isInitialized())
		{
			pSet->initialize();
		}

		// Create modules
		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_smoothingLength.connect(m_nbrQuery->m_radius);
		m_position.connect(m_nbrQuery->m_position);
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		m_pbdModule = std::make_shared<DensityPBD<TDataType>>();
		m_smoothingLength.connect(m_pbdModule->m_smoothingLength);
		m_position.connect(m_pbdModule->m_position);
		m_velocity.connect(m_pbdModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_pbdModule->m_neighborhood);
		m_pbdModule->initialize();

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		m_position.connect(m_integrator->m_position);
		m_velocity.connect(m_integrator->m_velocity);
		m_forceDensity.connect(m_integrator->m_forceDensity);
		m_integrator->initialize();

		m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->setViscosity(Real(1));
		m_smoothingLength.connect(m_visModule->m_smoothingLength);
		m_position.connect(m_visModule->m_position);
		m_velocity.connect(m_visModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_visModule->m_neighborhood);
		m_visModule->initialize();

		m_nbrQuery->setParent(parent);
		m_integrator->setParent(parent);
		m_pbdModule->setParent(parent);
		m_visModule->setParent(parent);
// 
// 		m_mapping = std::make_shared<PointSetToPointSet<TDataType>>();
// 		m_mapping->initialize(*(m_position.getReference()), (pSet->getPoints()));

		return true;
	}

	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}
		m_integrator->begin();

		m_nbrQuery->compute();

		auto& forceList = parent->getForceModuleList();
		auto fIter = forceList.begin();
		for (; fIter != forceList.end(); fIter++)
		{
			(*fIter)->applyForce();
		}

		m_integrator->integrate();
		
		m_pbdModule->constrain();

		m_visModule->constrain();
		auto& clist = parent->getConstraintModuleList();
		auto cIter = clist.begin();
		for (; cIter != clist.end(); cIter++)
		{
			(*cIter)->constrain();
		}
		
		m_integrator->end();
	}

	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_incompressibilitySolver)
		{
			getParent()->deleteConstraintModule(m_incompressibilitySolver);
		}
		m_incompressibilitySolver = solver;
		getParent()->addConstraintModule(solver);
	}


	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setViscositySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_viscositySolver)
		{
			getParent()->deleteConstraintModule(m_viscositySolver);
		}
		m_viscositySolver = solver;
		getParent()->addConstraintModule(solver);
	}



	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setSurfaceTensionSolver(std::shared_ptr<ForceModule> solver)
	{
		if (!m_surfaceTensionSolver)
		{
			getParent()->deleteForceModule(m_surfaceTensionSolver);
		}
		m_surfaceTensionSolver = solver;
		getParent()->addForceModule(m_surfaceTensionSolver);
	}

}