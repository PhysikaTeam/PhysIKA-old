#include "MultifluidModel.h"

#include <cuda_runtime.h>

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticleIntegrator.h"
#include "DensitySummation.h"
#include "ImplicitViscosity.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Mapping/PointSetToPointSet.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Framework/Topology/NeighborQuery.h"
#include "Physika_Dynamics/ParticleSystem/Attribute.h"
#include "Physika_Core/Utilities/cuda_utilities.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(MultifluidModel, TDataType)

    template <typename Real, typename Coord>
    __global__ void UpdateColor(DeviceArray<Vector3f> colorArr,
                                DeviceArray<Coord> posArr) {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
		colorArr[pId] = posArr[pId];
		colorArr[pId][0] = posArr[pId][0] > Real(0.5) ? 1 : 0;
	}
    template <typename Real, typename Coord>
    __global__ void Init(DeviceArray<Coord> posArr,
						 DeviceArray<Vector3f> colorArr,
						 DeviceArray<Real> massInvArr) {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
		if (posArr[pId][0] > 0.5) {
			colorArr[pId] = {1, 0, 0};
			massInvArr[pId] = 1;
		} else {
			colorArr[pId] = {0, 1, 0};
			massInvArr[pId] = 5;
		}
    }

    template<typename TDataType>
	MultifluidModel<TDataType>::MultifluidModel()
		: NumericalModel()
		, m_pNum(0)
	{
		m_smoothingLength.setValue(Real(0.044));

		initField(&m_smoothingLength, "smoothingLength", "Smoothing length", false);

		initField(&m_position, "position", "Storing the particle positions!", false);
		initField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		initField(&m_forceDensity, "force_density", "Storing the particle force densities!", false);
		initField(&m_massInv, "mass_inverse", "Storing mass inverse of particles!", false);
	}

	template<typename TDataType>
	MultifluidModel<TDataType>::~MultifluidModel()
	{
		
	}

	template<typename TDataType>
	bool MultifluidModel<TDataType>::initializeImpl()
	{
		this->NumericalModel::initializeImpl();

		m_massInv.setElementCount(m_position.getElementCount());
        int num = m_position.getElementCount();
        uint pDims = cudaGridSize(num, BLOCK_SIZE);
        Init<Real, Coord>
            <<<pDims, BLOCK_SIZE>>>(m_position.getValue(), m_color.getValue(), m_massInv.getValue());


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
		m_massInv.connect(m_pbdModule->m_massInv);
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
	void MultifluidModel<TDataType>::step(Real dt)
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
		// int num = m_position.getElementCount();
		// uint pDims = cudaGridSize(num, BLOCK_SIZE);
		// UpdateColor<Real, Coord><<<pDims, BLOCK_SIZE>>>(
        //     m_color.getValue(), m_position.getValue());
	}

	template<typename TDataType>
	void MultifluidModel<TDataType>::setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_incompressibilitySolver)
		{
			getParent()->deleteConstraintModule(m_incompressibilitySolver);
		}
		m_incompressibilitySolver = solver;
		getParent()->addConstraintModule(solver);
	}


	template<typename TDataType>
	void MultifluidModel<TDataType>::setViscositySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_viscositySolver)
		{
			getParent()->deleteConstraintModule(m_viscositySolver);
		}
		m_viscositySolver = solver;
		getParent()->addConstraintModule(solver);
	}



	template<typename TDataType>
	void MultifluidModel<TDataType>::setSurfaceTensionSolver(std::shared_ptr<ForceModule> solver)
	{
		if (!m_surfaceTensionSolver)
		{
			getParent()->deleteForceModule(m_surfaceTensionSolver);
		}
		m_surfaceTensionSolver = solver;
		getParent()->addForceModule(m_surfaceTensionSolver);
	}

}