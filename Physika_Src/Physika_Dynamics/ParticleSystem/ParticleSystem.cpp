#include "ParticleSystem.h"
#include "Framework/DeviceContext.h"
#include "Attribute.h"
#include "INeighbors.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticlePrediction.h"
#include "NeighborQuery.h"
#include "SummationDensity.h"
#include "BoundaryManager.h"
#include "ViscosityBase.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
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
	bool ParticleSystem<TDataType>::initialize()
	{
		this->NumericalModel::initialize();

		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		PointSet<Coord>* pSet = dynamic_cast<PointSet<Coord>*>(parent->getTopologyModule());
		if (pSet == NULL)
		{
			Log::sendMessage(Log::Error, "The topology module is not supported!");
			return false;
		}

		m_num = this->allocHostVariable<size_t>("num", "Particle number", (size_t)pSet->getPointSize());
		m_mass = this->allocHostVariable<Real>("mass", "Particle mass", Real(1));
		m_smoothingLength = this->allocHostVariable<Real>("smoothingLength", "Smoothing length", Real(0.0125));
		m_samplingDistance = this->allocHostVariable<Real>("samplingDistance", "Sampling distance", Real(0.005));
		m_restDensity = this->allocHostVariable<Real>("restDensity", "Rest density", Real(1000));

		m_lowerBound = this->allocHostVariable<Coord>("lowerBound", "Lower bound", Coord(0));
		m_upperBound = this->allocHostVariable<Coord>("upperBound", "Upper bound", Coord(1));

		m_gravity = this->allocHostVariable<Coord>("gravity", "gravity", Coord(0.0f, -9.8f, 0.0f));

		std::shared_ptr<DeviceContext> dc = getParent()->getContext();

		dc->enable();

		std::cout << "Point number: " << m_num->getValue() << std::endl;
		auto posBuf = dc->allocDeviceBuffer<Coord>("POSITION1", "Particle positions", m_num->getValue());
		auto velBuf = dc->allocDeviceBuffer<Coord>("VELOCITY1", "Particle velocities", m_num->getValue());
		auto restPos = dc->allocDeviceBuffer<Coord>("OLD_POSITION1", "Old particle positions", m_num->getValue());
		auto restVel = dc->allocDeviceBuffer<Coord>("OLD_VELOCITY1", "Particle positions", m_num->getValue());
		auto rhoBuf = dc->allocDeviceBuffer<Real>("DENSITY1", "Particle densities", m_num->getValue());
		auto neighborBuf = dc->allocDeviceBuffer<SPHNeighborList>("NEIGHBORHOOD1", "Particle neighbor ids", m_num->getValue());
		auto attBuf = dc->allocDeviceBuffer<Attribute>("ATTRIBUTE1", "Particle attributes", m_num->getValue());

		auto pbdModule = new DensityPBD<TDataType>();
		pbdModule->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		pbdModule->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		pbdModule->connectDensity(TypeInfo::CastPointerUp<Field>(rhoBuf));
		pbdModule->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
		pbdModule->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));
		pbdModule->connectMass(TypeInfo::CastPointerUp<Field>(m_mass));
		pbdModule->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));

		auto prediction = new ParticlePrediction<TDataType>();
		prediction->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		prediction->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		prediction->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));

		auto nQuery = new NeighborQuery<TDataType>();
		nQuery->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		nQuery->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));
		nQuery->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		nQuery->connectSmoothingLength(TypeInfo::CastPointerUp<Field>(m_smoothingLength));

		auto* bmgr = new BoundaryManager<TDataType>();
		Coord lo(0.0f);
		Coord hi(1.0f);
		DistanceField3D<TDataType> * box = new DistanceField3D<TDataType>();
		box->SetSpace(lo - m_samplingDistance->getValue() * 5, hi + m_samplingDistance->getValue() * 5, 105, 105, 105);
		box->DistanceFieldToBox(lo, hi, true);
		//		box->DistanceFieldToSphere(make_float3(0.5f), 0.2f, true);
		bmgr->InsertBarrier(new BarrierDistanceField3D<TDataType>(box));
		bmgr->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));
		bmgr->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		bmgr->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));

		auto* visModule = new ViscosityBase<TDataType>();
		visModule->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		visModule->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		visModule->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));
		visModule->connectDensity(TypeInfo::CastPointerUp<Field>(rhoBuf));
		visModule->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		visModule->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
		visModule->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));

		parent->addModule(nQuery);
		parent->addModule(pbdModule);
		parent->addModule(prediction);
		parent->addModule(bmgr);
		parent->addModule(visModule);
		
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
			summation->initialize();
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

		return true;
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::execute()
	{
		PointSet<Coord>* pSet = dynamic_cast<PointSet<Coord>*>(getParent()->getTopologyModule());
		std::shared_ptr<DeviceContext> dc = getParent()->getContext();
		dc->enable();
		std::shared_ptr<Field> field = dc->getField("POSITION1");
		std::shared_ptr<DeviceBuffer<Coord>> pBuf = TypeInfo::CastPointerDown<DeviceBuffer<Coord>>(field);

		Function1Pt::Copy(*(pSet->getPoints()), *(pBuf->getDataPtr()));
		return true;
	}


}