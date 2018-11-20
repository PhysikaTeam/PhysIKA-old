#include "Peridynamics.h"
#include "Physika_Core/Utilities/Reduction.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "NeighborQuery.h"
#include "Physika_Framework/Framework/DeviceContext.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Topology/PointSet.h"

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

		m_num = HostVarField<size_t>::createField(mstate.get(), "num", "Particle number", (size_t)pSet->getPointSize());
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
		auto rhoBuf = DeviceArrayField<Real>::createField(mstate.get(), "DENSITY1", "Particle densities", m_num->getValue());
		auto neighborBuf = DeviceArrayField<SPHNeighborList>::createField(mstate.get(), "NEIGHBORHOOD1", "Particle neighbor ids", m_num->getValue());
		auto restShapeBuf = DeviceArrayField<RestShape>::createField(mstate.get(), "RESTSHAPE", "Particle neighbor ids", m_num->getValue());
		auto stateBuf = DeviceArrayField<int>::createField(mstate.get(), "State", "Particle states", m_num->getValue());
		auto initPosBuf = DeviceArrayField<Coord>::createField(mstate.get(), MechanicalState::init_position(), "Initial particle positions", m_num->getValue());
		auto attBuf = DeviceArrayField<Attribute>::createField(mstate.get(), "ATTRIBUTE1", "Particle attributes", m_num->getValue());

		prediction = std::make_shared<ParticlePrediction<TDataType>>();
		prediction->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		prediction->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		prediction->connectAttribute(TypeInfo::CastPointerUp<Field>(attBuf));

		auto nQuery = std::make_shared<NeighborQuery<TDataType>>();
		nQuery->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		nQuery->connectNeighbor(TypeInfo::CastPointerUp<Field>(neighborBuf));
		nQuery->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		nQuery->connectSmoothingLength(TypeInfo::CastPointerUp<Field>(m_smoothingLength));

		elasticity = std::make_shared<ElasticityModule<TDataType>>();
		elasticity->connectPosition(TypeInfo::CastPointerUp<Field>(posBuf));
		elasticity->connectVelocity(TypeInfo::CastPointerUp<Field>(velBuf));
		elasticity->connectRadius(TypeInfo::CastPointerUp<Field>(m_smoothingLength));
		elasticity->connectSamplingDistance(TypeInfo::CastPointerUp<Field>(m_samplingDistance));
		elasticity->connectState(TypeInfo::CastPointerUp<Field>(stateBuf));
		elasticity->connectPrePosition(TypeInfo::CastPointerUp<Field>(prePos));
		elasticity->connectInitPosition(TypeInfo::CastPointerUp<Field>(initPosBuf));
		elasticity->connectRestShape(TypeInfo::CastPointerUp<Field>(restShapeBuf));

		parent->addModule(nQuery);
		parent->addModule(prediction);
		parent->addModule(elasticity);


		std::vector<Coord> positions;
		std::vector<Coord> velocities;
		std::vector<Attribute> attributes;
		std::vector<int> states;
		HostArray<Coord>* hPosArr = new HostArray<Coord>(m_num->getValue());
		Function1Pt::Copy(*(hPosArr), *(pSet->getPoints()));

		for (int i = 0; i < m_num->getValue(); i++)
		{
			velocities.push_back(Coord(0, 0, 0));
			Attribute attri;
			attri.SetFluid();
			attri.SetDynamic();
			attributes.push_back(attri);
			if ((*hPosArr)[i][0] < 0.4075)
			{
				states.push_back(1);
			}
			else
			{
				states.push_back(0);
			}
		}
		Function1Pt::Copy(*(posBuf->getDataPtr()), *(pSet->getPoints()));
		Function1Pt::Copy(*(velBuf->getDataPtr()), velocities);
		Function1Pt::Copy(*(attBuf->getDataPtr()), attributes);
		Function1Pt::Copy(*(stateBuf->getDataPtr()), states);
		Function1Pt::Copy(*(initPosBuf->getDataPtr()), *(pSet->getPoints()));

		nQuery->execute();

		HostArray<SPHNeighborList>* hostNeighbor = new HostArray<SPHNeighborList>(m_num->getValue());
		HostArray<RestShape>* restNeighbor = new HostArray<RestShape>(m_num->getValue());
		HostArray<Coord>* hostInitPos = new HostArray<Coord>(m_num->getValue());

		DeviceArray<SPHNeighborList>* deviceNeighbor = neighborBuf->getDataPtr();
		DeviceArray<RestShape>* deviceRestNeighbor = restShapeBuf->getDataPtr();
		DeviceArray<Coord>* deviceInitPos = initPosBuf->getDataPtr();

		Function1Pt::Copy(*hostNeighbor, *deviceNeighbor);
		Function1Pt::Copy(*hostInitPos, *deviceInitPos);

		for (int i = 0; i < m_num->getValue(); i++)
		{
			SPHNeighborList& neighborlist_i = (*hostNeighbor)[i];
			RestShape& rest_i = (*restNeighbor)[i];
			int size_i = neighborlist_i.size;
			rest_i.size = size_i;
			rest_i.idx = UNDEFINED;
			for (int ne = 0; ne < size_i; ne++)
			{
				int j = neighborlist_i.ids[ne];
				rest_i.ids[ne] = j;
				rest_i.distance[ne] = ((*hostInitPos)[i] - (*hostInitPos)[j]).norm();
				rest_i.pos[ne] = (*hostInitPos)[j];
				if (j == i)
				{
					rest_i.idx = ne;
//					std::cout << rest_i.distance[ne] << std::endl;
				}
			}


		}

		Function1Pt::Copy(*deviceRestNeighbor, *restNeighbor);


// 		HostArray<RestShape>* hRestShape = restNeighbor;// new HostArray<RestShape>(restNeighbor->size());
// 		Function1Pt::Copy(*hRestShape, *deviceRestNeighbor);
// 		for (int i = 0; i < hRestShape->size(); i++)
// 		{
// 			SPHNeighborList& neighborlist_i = (*hostNeighbor)[i];
// 			RestShape si = (*hRestShape)[i];
// 			Coord rest_i = (*hRestShape)[i].pos[(*hRestShape)[i].idx];
// 			int size_i = (*hRestShape)[i].size;
// 
// 			for (int ne = 0; ne < size_i; ne++)
//			{
// 				int j = (*hRestShape)[i].ids[ne];
// 				Real r = (*hRestShape)[i].distance[ne];
// 
// 				if (r > EPSILON)
// 				{
// 					Coord rest_j = (*hRestShape)[i].pos[ne];
// 					Coord q = (rest_i - (*hRestShape)[i].pos[ne])*(1.0f / r);
// 					if (q.norm() < EPSILON)
// 					{
// 						printf("%d %d %f %f %f %f %f %f \n", i, (*hRestShape)[i].ids[ne], rest_i[0], rest_i[1], rest_i[2], (*hRestShape)[i].pos[ne][0], (*hRestShape)[i].pos[ne][1], (*hRestShape)[i].pos[ne][2]);
// 					}
// 				}
// 			}
// 		}

		hostNeighbor->release();
		restNeighbor->release();
		delete hostNeighbor;
		delete restNeighbor;

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

		Function1Pt::Copy(*preBuf, *posBuf);

// 		auto& list = parent->getModuleList();
// 		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
// 		for (; iter != list.end(); iter++)
// 		{
// 			(*iter)->execute();
// 		}

		prediction->execute();
		elasticity->execute();

 		auto& list = parent->getConstraintModuleList();
 		std::list<std::shared_ptr<ConstraintModule>>::iterator iter = list.begin();
 		for (; iter != list.end(); iter++)
 		{
			(*iter)->constrain();
 		}
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