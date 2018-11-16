#include "CollidablePoints.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/DeviceContext.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Mapping/RigidToPoints.h"
#include "Physika_Framework/Mapping/PointsToPoints.h"
#include "Physika_Core/Utilities/Reduction.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(CollidablePoints, TDataType)

	template<typename TDataType>
	CollidablePoints<TDataType>::CollidablePoints()
		: CollidableObject(CollidableObject::POINTSET_TYPE)
		, m_bUniformRaidus(true)
		, m_radius(0.005)
	{
	}
	
	template<typename TDataType>
	CollidablePoints<TDataType>::~CollidablePoints()
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setPositions(DeviceArray<Coord>& centers)
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setVelocities(DeviceArray<Coord>& vel)
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setRadii(DeviceArray<Coord>& radii)
	{

	}

	template<typename TDataType>
	bool Physika::CollidablePoints<TDataType>::initializeImpl()
	{
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

		auto initPoints = pSet->getPoints();

		m_initPos.resize(initPoints->size());
		m_positions.resize(initPoints->size());
		Function1Pt::Copy(m_positions, *initPoints);

		m_velocities.resize(initPoints->size());
		m_velocities.reset();

		auto mstate = getParent()->getMechanicalState();
		auto mType = getParent()->getMechanicalState()->getMaterialType();

		if (mType == MechanicalState::RIGIDBODY)
		{
			auto mapping = std::make_shared<RigidToPoints<TDataType>>();
			auto center = mstate->getField<HostVariable<Coord>>(MechanicalState::position())->getValue();
			auto rotation = mstate->getField<HostVariable<Matrix>>(MechanicalState::rotation())->getValue();

			mapping->initialize(Rigid(center, Quaternion<Real>(rotation)), m_positions);
			m_mapping = mapping;
		}
		else
		{
			auto mapping = std::shared_ptr<PointsToPoints<TDataType>>();
			m_mapping = mapping;
		}
	}


	template<typename TDataType>
	void CollidablePoints<TDataType>::updateCollidableObject()
	{
		auto mstate = getParent()->getMechanicalState();
		auto mType = mstate->getMaterialType();
		if (mType == MechanicalState::RIGIDBODY)
		{
			auto center = mstate->getField<HostVariable<Coord>>(MechanicalState::position())->getValue();
			auto rotation = mstate->getField<HostVariable<Matrix>>(MechanicalState::rotation())->getValue();

			auto pSet = TypeInfo::CastPointerDown<PointSet<TDataType>>(getParent()->getTopologyModule());

			auto mp = std::dynamic_pointer_cast<RigidToPoints<TDataType>>(m_mapping);

			mp->applyTransform(Rigid(center, Quaternion<Real>(rotation)), m_positions);

		}
		else
		{
			std::shared_ptr<Field> pos = mstate->getField(MechanicalState::position());
			std::shared_ptr<DeviceBuffer<Coord>> pBuf = TypeInfo::CastPointerDown<DeviceBuffer<Coord>>(pos);

			std::shared_ptr<Field> vel = mstate->getField(MechanicalState::velocity());
			std::shared_ptr<DeviceBuffer<Coord>> vBuf = TypeInfo::CastPointerDown<DeviceBuffer<Coord>>(vel);

			Function1Pt::Copy(m_positions, *(pBuf->getDataPtr()));
			Function1Pt::Copy(m_velocities, *(vBuf->getDataPtr()));
		}

		Function1Pt::Copy(m_initPos, m_positions);
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::updateMechanicalState()
	{
		auto mstate = getParent()->getMechanicalState();
		auto mType = mstate->getMaterialType();
		auto dc = getParent()->getMechanicalState();
		if (mType == MechanicalState::RIGIDBODY)
		{
			HostArray<Coord> hPos;
			HostArray<Coord> hInitPos;
			hPos.resize(m_positions.size());
			hInitPos.resize(m_positions.size());
			Function1Pt::Copy(hPos, m_positions);
			Function1Pt::Copy(hInitPos, m_initPos);
			Coord center(0);
			int nn = 0;
			for (int i = 0; i < hPos.size(); i++)
			{
				if ((hInitPos[i] - hPos[i]).norm() > EPSILON)
				{
					center += (hPos[i] - hInitPos[i]);
					nn++;
				}
					
			}
			if(nn > 0)
				center /= nn;

			auto massCenter = mstate->getField<HostVariable<Coord>>(MechanicalState::position())->getValue();
			dc->getField<HostVariable<Coord>>(MechanicalState::position())->setValue(center+ massCenter);

			hPos.release();
		}
		else
		{
			auto posArr = dc->getField<DeviceBuffer<Coord>>(MechanicalState::position());
			auto velArr = dc->getField<DeviceBuffer<Coord>>(MechanicalState::velocity());

			Function1Pt::Copy(*(posArr->getDataPtr()), m_positions);
			Function1Pt::Copy(*(velArr->getDataPtr()), m_velocities);
		}
	}
}