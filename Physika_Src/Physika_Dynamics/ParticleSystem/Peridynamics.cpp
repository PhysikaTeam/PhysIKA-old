#include "Peridynamics.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/DeviceContext.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Mapping/PointsToPoints.h"
#include "ParticleIntegrator.h"
#include "Physika_Framework/Topology/NeighborQuery.h"
#include "HyperelasticForce.h"

namespace Physika 
{
	IMPLEMENT_CLASS_1(Peridynamics, TDataType)

	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: NumericalModel()
	{
	}

	template<typename TDataType>
	bool Peridynamics<TDataType>::initializeImpl()
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

		auto mstate = getParent()->getMechanicalState();
		mstate->setMaterialType(MechanicalState::ELASTIC);

		m_num = HostVarField<int>::createField(mstate.get(), "num", "Particle number", pSet->getPointSize());
		m_mass = HostVarField<Real>::createField(mstate.get(), MechanicalState::mass(), "Particle mass", Real(1));
		m_smoothingLength = HostVarField<Real>::createField(mstate.get(), "smoothingLength", "Smoothing length", Real(0.0125));
		m_samplingDistance = HostVarField<Real>::createField(mstate.get(), "samplingDistance", "Sampling distance", Real(0.005));
		m_restDensity = HostVarField<Real>::createField(mstate.get(), "restDensity", "Rest density", Real(1000));

		std::cout << "Particle Number: " << m_num->getValue() << std::endl;

		Real d = m_samplingDistance->getValue();
		Real rho = m_restDensity->getValue();
		m_mass->setValue(rho*d*d*d);

		auto posBuf = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::position(), "Particle positions", m_num->getValue());
		auto velBuf = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::velocity(), "Particle velocities", m_num->getValue());
		auto prePos = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::pre_position(), "Old particle positions", m_num->getValue());
		auto preVel = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::pre_velocity(), "Particle positions", m_num->getValue());
		auto force = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::force(), "Particle forces", m_num->getValue());
		auto initPosBuf = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::init_position(), "Initial particle positions", m_num->getValue());
		auto nbrBuf = NeighborField<int>::createField(mstate.get(), MechanicalState::particle_neighbors(), "Reference particles", m_num->getValue(), 30);
		auto refNpos = NeighborField<Coord>::createField(mstate.get(), MechanicalState::reference_particles(), "Reference particles", m_num->getValue(), 30);

		prediction = std::make_shared<ParticleIntegrator<TDataType>>();

		auto nQuery = std::make_shared<NeighborQuery<TDataType>>();
		nQuery->setRadius(0.0125);

		m_elasticity = std::make_shared<ElasticityModule<TDataType>>();
		m_elasticity->setHorizon(0.0125);

		parent->addModule(nQuery);
		parent->addModule(prediction);
		parent->addConstraintModule(m_elasticity);

		Function1Pt::copy(*(posBuf->getDataPtr()), *(pSet->getPoints()));
		Function1Pt::copy(*(initPosBuf->getDataPtr()), *(pSet->getPoints()));

		nQuery->compute();

		m_mapping = std::make_shared<PointsToPoints<TDataType>>();
		m_mapping->initialize(*(posBuf->getDataPtr()), *(pSet->getPoints()));

		return true;
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}

		auto dc = getParent()->getMechanicalState();
		auto posBuf = dc->getField<DeviceArrayField<Coord>>(MechanicalState::position())->getDataPtr();
		auto preBuf = dc->getField<DeviceArrayField<Coord>>(MechanicalState::pre_position())->getDataPtr();

		Function1Pt::copy(*preBuf, *posBuf);

		prediction->begin();

		auto& forceList = parent->getForceModuleList();
		auto fIter = forceList.begin();
		for (; fIter != forceList.end(); fIter++)
		{
			(*fIter)->applyForce();
		}

		prediction->integrate();

 		auto& constraintList = parent->getConstraintModuleList();
 		std::list<std::shared_ptr<ConstraintModule>>::reverse_iterator iter = constraintList.rbegin();
 		for (; iter != constraintList.rend(); iter++)
 		{
			(*iter)->constrain();
 		}

		prediction->end();
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::updateTopology()
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<TDataType>>(getParent()->getTopologyModule());
		auto dc = getParent()->getMechanicalState();
		std::shared_ptr<Field> field = dc->getField(MechanicalState::position());
		std::shared_ptr<DeviceArrayField<Coord>> pBuf = TypeInfo::CastPointerDown<DeviceArrayField<Coord>>(field);

		m_mapping->applyTranform(*(pBuf->getDataPtr()), *(pSet->getPoints()));
		//Function1Pt::Copy(*(pSet->getPoints()), *(pBuf->getDataPtr()));
	}
}