#include "ParticleFluid.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticleIntegrator.h"
#include "DensitySummation.h"
#include "ImplicitViscosity.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Mapping/PointsToPoints.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Framework/Topology/NeighborQuery.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid()
		: NumericalModel()
		, m_smoothingL(Real(0.0125))
		, m_samplingD(Real(0.005))
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{

	}

	template<typename TDataType>
	bool ParticleFluid<TDataType>::initializeImpl()
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

		auto mstate = parent->getMechanicalState();
		mstate->setMaterialType(MechanicalState::FLUID);

		m_pNum = pSet->getPointSize();
		std::cout << "Point number: " << m_pNum << std::endl;
		
		m_smoothingLength = HostVarField<Real>::createField(this, "smoothingLength", "Smoothing length", Real(0.011));
		m_samplingDistance = HostVarField<Real>::createField(this, "samplingDistance", "Sampling distance", Real(0.005));
		m_restDensity = HostVarField<Real>::createField(this, "restDensity", "Rest density", Real(1000));

		m_lowerBound = HostVarField<Coord>::createField(this, "lowerBound", "Lower bound", Coord(0));
		m_upperBound = HostVarField<Coord>::createField(this, "upperBound", "Upper bound", Coord(1));

		// Allocate mechanical states
		auto m_mass		= HostVarField<Real>::createField(mstate.get(), MechanicalState::mass(), "Particle mass", Real(1));
		auto posBuf		= DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::position(), "Particle positions", m_pNum);
		auto velBuf		= DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::velocity(), "Particle velocities", m_pNum);
		auto restPos	= DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::pre_position(), "Old particle positions", m_pNum);
		auto force		= DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::force(), "Particle forces", m_pNum);
		auto restVel	= DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::pre_velocity(), "Particle positions", m_pNum);
		auto rhoBuf		= DeviceArrayField<Real>::createField(mstate.get(), MechanicalState::density(), "Particle densities", m_pNum);
		auto adaptNbr	= NeighborField<int>::createField(mstate.get(), MechanicalState::particle_neighbors(), "Particle neighbor ids", m_pNum);

		// Create modules
		auto pbdModule = std::make_shared<DensityPBD<TDataType>>();
		pbdModule->setSmoothingLength(m_smoothingL);

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();

		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_nbrQuery->setRadius(m_smoothingL);

		auto visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		visModule->setSmoothingLength(m_smoothingL);
		visModule->setViscosity(Real(1));

		parent->addModule(m_nbrQuery);
		parent->addModule(m_integrator);
		parent->addConstraintModule(pbdModule);
		parent->addConstraintModule(visModule);

		m_nbrQuery->setRadius(m_smoothingLength->getValue());
		
		std::vector<Coord> positions;
		std::vector<Coord> velocities;
		for (int i = 0; i < m_pNum; i++)
		{
			velocities.push_back(Coord(0, 0.0, 0));
		}
		Function1Pt::copy(*(posBuf->getDataPtr()), *(pSet->getPoints()));
		Function1Pt::copy(*(velBuf->getDataPtr()), velocities);

 		if (1)
 		{
			m_nbrQuery->compute();
			auto summation = std::make_shared<DensitySummation<TDataType>>();
			summation->setSmoothingLength(m_smoothingL);
			parent->addModule(summation);
			summation->compute();
			//parent->deleteModule(summation);

			DeviceArray<Real>* gpgRho = rhoBuf->getDataPtr();

			Reduction<Real>* pReduce = Reduction<Real>::Create(gpgRho->size());

			Real maxRho = pReduce->Maximum(gpgRho->getDataPtr(), gpgRho->size());

			Real newMass = m_restDensity->getValue() / maxRho * m_mass->getValue();
			m_mass->setValue(newMass);

			std::cout << "Test for Maximum Density: " << maxRho << std::endl;

			delete pReduce;
 		}

		m_mapping = std::make_shared<PointsToPoints<TDataType>>();
		m_mapping->initialize(*(posBuf->getDataPtr()), *(pSet->getPoints()));

		return true;
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::step(Real dt)
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

		auto& clist = parent->getConstraintModuleList();
		auto cIter = clist.begin();
		for (; cIter != clist.end(); cIter++)
		{
			(*cIter)->constrain();
		}

		m_integrator->end();
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::updateTopology()
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<TDataType>>(getParent()->getTopologyModule());
		auto dc = getParent()->getMechanicalState();
		auto pBuf = dc->getField<DeviceArrayField<Coord>>(MechanicalState::position());

		m_mapping->applyTranform(*(pBuf->getDataPtr()), *(pSet->getPoints()));
		//Function1Pt::Copy(*(pSet->getPoints()), *(pBuf->getDataPtr()));
	}
}