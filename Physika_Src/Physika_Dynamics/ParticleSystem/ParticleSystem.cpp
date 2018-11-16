#include "ParticleSystem.h"
#include "Physika_Framework/Framework/DeviceContext.h"
#include "Attribute.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticlePrediction.h"
#include "NeighborQuery.h"
#include "SummationDensity.h"
#include "ViscosityBase.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/MechanicalState.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleSystem, TDataType)

	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem()
		: NumericalModel()
	{
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{

	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::initializeImpl()
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

		m_num = HostVariable<size_t>::createField(mstate.get(), "num", "Particle number", (size_t)pSet->getPointSize());
		m_mass = HostVariable<Real>::createField(mstate.get(), MechanicalState::mass(), "Particle mass", Real(1));
		m_smoothingLength = HostVariable<Real>::createField(mstate.get(), "smoothingLength", "Smoothing length", Real(0.0125));
		m_samplingDistance = HostVariable<Real>::createField(mstate.get(), "samplingDistance", "Sampling distance", Real(0.005));
		m_restDensity = HostVariable<Real>::createField(mstate.get(), "restDensity", "Rest density", Real(1000));

		m_lowerBound = HostVariable<Coord>::createField(mstate.get(), "lowerBound", "Lower bound", Coord(0));
		m_upperBound = HostVariable<Coord>::createField(mstate.get(), "upperBound", "Upper bound", Coord(1));

		m_gravity = HostVariable<Coord>::createField(mstate.get(),"gravity", "gravity", Coord(0.0f, -9.8f, 0.0f));

		std::shared_ptr<DeviceContext> dc = getParent()->getContext();

		dc->enable();

		std::cout << "Point number: " << m_num->getValue() << std::endl;
		auto posBuf = DeviceBuffer<Coord>::createField(mstate.get(), MechanicalState::position(), "Particle positions", m_num->getValue());
		auto velBuf = DeviceBuffer<Coord>::createField(mstate.get(), MechanicalState::velocity(), "Particle velocities", m_num->getValue());
		auto restPos = DeviceBuffer<Coord>::createField(mstate.get(), MechanicalState::pre_position(), "Old particle positions", m_num->getValue());
		auto restVel = DeviceBuffer<Coord>::createField(mstate.get(), MechanicalState::pre_velocity(), "Particle positions", m_num->getValue());
		auto rhoBuf = DeviceBuffer<Real>::createField(mstate.get(), "DENSITY1", "Particle densities", m_num->getValue());
		auto neighborBuf = DeviceBuffer<SPHNeighborList>::createField(mstate.get(), "NEIGHBORHOOD1", "Particle neighbor ids", m_num->getValue());
		auto attBuf = DeviceBuffer<Attribute>::createField(mstate.get(), "ATTRIBUTE1", "Particle attributes", m_num->getValue());

		auto pbdModule = std::make_shared<DensityPBD<TDataType>>();
		pbdModule->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		pbdModule->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		pbdModule->connectDensity(TypeInfo::CastPointerUp<Field>(rhoBuf));
		pbdModule->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
		pbdModule->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));
		pbdModule->connectMass(TypeInfo::CastPointerUp<Field>(m_mass));
		pbdModule->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));

		auto prediction = std::make_shared<ParticlePrediction<TDataType>>();
		prediction->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		prediction->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		prediction->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));

		auto nQuery = std::make_shared<NeighborQuery<TDataType>>();
		nQuery->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		nQuery->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));
		nQuery->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		nQuery->connectSmoothingLength(TypeInfo::CastPointerUp<Field>(m_smoothingLength));

		auto visModule = std::make_shared<ViscosityBase<TDataType>>();
		visModule->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		visModule->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		visModule->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));
		visModule->connectDensity(TypeInfo::CastPointerUp<Field>(rhoBuf));
		visModule->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		visModule->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
		visModule->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));

		parent->addModule(nQuery);
		//parent->addModule(pbdModule);
		parent->addConstraintModule(pbdModule);
		parent->addModule(prediction);
//		parent->addModule(bmgr);
		//parent->addConstraintModule(bmgr);
		//parent->addModule(visModule);
		parent->addForceModule(visModule);
		
		std::vector<Coord> positions;
		std::vector<Coord> velocities;
		std::vector<Attribute> attributes;
		for (int i = 0; i < m_num->getValue(); i++)
		{
			velocities.push_back(Coord(0, -1.0, 0));
			Attribute attri;
			attri.SetFluid();
			attri.SetDynamic();
			attributes.push_back(attri);
		}
		Function1Pt::Copy(*(posBuf->getDataPtr()), *(pSet->getPoints()));
		Function1Pt::Copy(*(velBuf->getDataPtr()), velocities);
		Function1Pt::Copy(*(attBuf->getDataPtr()), attributes);


 		if (1)
 		{
 			nQuery->execute();
			auto summation = new SummationDensity<TDataType>();
			summation->setParent(parent);
			summation->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
			summation->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));
			summation->connectMass(TypeInfo::CastPointerUp<Field>(m_mass));
			summation->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
			summation->connectDensity(TypeInfo::CastPointerUp<Field>(rhoBuf));
			summation->initializeImpl();
			summation->execute();

			DeviceArray<Real>* gpgRho = rhoBuf->getDataPtr();

			Reduction<Real>* pReduce = Reduction<Real>::Create(gpgRho->size());

			Real maxRho = pReduce->Maximum(gpgRho->getDataPtr(), gpgRho->size());

			Real newMass = m_restDensity->getValue() / maxRho * m_mass->getValue();
			m_mass->setValue(newMass);

			std::cout << "Test for Maximum Density: " << maxRho << std::endl;

			delete pReduce;
 		}
// 
// 		Function1Pt::Copy(*(pSet->getPoints()), *(posBuf->getDataPtr()));
// 		BoundaryManager<TDataType>* bmgr = new BoundaryManager<TDataType>(this);
// 
// 		DistanceField3D<TDataType> * box = new DistanceField3D<TDataType>();
// 		box->SetSpace(this->GetLowerBound() - this->GetSamplingDistance() * 5, this->GetUpperBound() + this->GetSamplingDistance() * 5, 105, 105, 105);
// 		box->DistanceFieldToBox(this->GetLowerBound(), this->GetUpperBound(), true);
// 		//		box->DistanceFieldToSphere(make_float3(0.5f), 0.2f, true);
// 		bmgr->InsertBarrier(new BarrierDistanceField3D<TDataType>(box));
// 		parent->addModule("BOUNDARY_HANDLING", bmgr);

		m_mapping = std::make_shared<PointsToPoints<TDataType>>();
		m_mapping->initialize(*(posBuf->getDataPtr()), *(pSet->getPoints()));

		return true;
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::execute()
	{
		return true;
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}
		auto& list = parent->getModuleList();
		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->execute();
		}
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::updateTopology()
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<TDataType>>(getParent()->getTopologyModule());
		auto dc = getParent()->getMechanicalState();
		auto pBuf = dc->getField<DeviceBuffer<Coord>>(MechanicalState::position());

		m_mapping->applyTranform(*(pBuf->getDataPtr()), *(pSet->getPoints()));
		//Function1Pt::Copy(*(pSet->getPoints()), *(pBuf->getDataPtr()));
	}
}