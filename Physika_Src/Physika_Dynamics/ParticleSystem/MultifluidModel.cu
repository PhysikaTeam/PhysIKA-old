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

	template <typename Real, typename Coord, typename PhaseVector>
	__global__ void UpdateMassInv(
		DeviceArray<Real> massInvArr,
		DeviceArray<PhaseVector> cArr,
		PhaseVector rho0) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= cArr.size()) return;
		Real rho = rho0.dot(cArr[pId]);
		massInvArr[pId] = Real(1)/rho;
	}
	template <typename Real, typename Coord, typename PhaseVector>
	__global__ void UpdateColor(DeviceArray<Vector3f> colorArr,
								DeviceArray<PhaseVector> cArr) {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= cArr.size()) return;
		auto c = cArr[pId];
		colorArr[pId] = {c[0],c[0],c[1]};
	}
    template <typename Real, typename Coord, typename PhaseVector>
    __global__ void InitConcentration(DeviceArray<Coord> posArr,
						 DeviceArray<PhaseVector> cArr) {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
		if (pId % 2 == 0) {
			cArr[pId] = {0.6, 0.4};
		} else {
			cArr[pId] = {0.4, 0.6};
		}
    }

    template<typename TDataType>
	MultifluidModel<TDataType>::MultifluidModel()
		: NumericalModel()
		, m_pNum(0)
	{
		m_smoothingLength.setValue(Real(0.044));
		m_restDensity.setValue(PhaseVector(1, 5));

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

		m_concentration.setElementCount(m_position.getElementCount());
		m_massInv.setElementCount(m_position.getElementCount());
        int num = m_position.getElementCount();
        uint pDims = cudaGridSize(num, BLOCK_SIZE);
        InitConcentration<Real, Coord, PhaseVector>
            <<<pDims, BLOCK_SIZE>>>(m_position.getValue(), m_concentration.getValue());


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

		m_phaseSolver = std::make_shared<CahnHilliard<TDataType>>();
		m_position.connect(m_phaseSolver->m_position);
		m_concentration.connect(m_phaseSolver->m_concentration);
		m_nbrQuery->m_neighborhood.connect(m_phaseSolver->m_neighborhood);
		m_smoothingLength.connect(m_phaseSolver->m_smoothingLength);
		m_phaseSolver->initialize();


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
		m_phaseSolver->setParent(parent);
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

		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		m_integrator->begin();

		m_nbrQuery->compute();

		auto& forceList = parent->getForceModuleList();
		auto fIter = forceList.begin();
		for (; fIter != forceList.end(); fIter++)
		{
			(*fIter)->applyForce();
		}

		m_integrator->integrate();

		m_phaseSolver->integrate();
		
		UpdateMassInv<Real, Coord, PhaseVector><<<pDims, BLOCK_SIZE>>>(
            m_massInv.getValue(), m_concentration.getValue(), m_restDensity.getValue());
		m_pbdModule->constrain();

		m_visModule->constrain();
		auto& clist = parent->getConstraintModuleList();
		auto cIter = clist.begin();
		for (; cIter != clist.end(); cIter++)
		{
			(*cIter)->constrain();
		}
		
		m_integrator->end();

		UpdateColor<Real, Coord><<<pDims, BLOCK_SIZE>>>(
            m_color.getValue(), m_concentration.getValue());
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