#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/CollisionModel.h"

namespace PhysIKA
{
template <typename> class CollidablePoints;
template <typename> class NeighborQuery;
template <typename> class NeighborList;
template <typename> class GridHash;

template<typename TDataType>
class RodCollision : public CollisionModel
{
	DECLARE_CLASS_1(RodCollision, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	RodCollision();
	virtual ~RodCollision();

	bool isSupport(std::shared_ptr<CollidableObject> obj) override;

	void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

	bool initializeImpl() override;

	void doCollision() override;
	
protected:
	DeviceArray<int> m_objId;
	DeviceArray<Coord> m_points;
	DeviceArray<Coord> m_vels;

	std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
	std::shared_ptr<NeighborList<int>> m_nList;

	std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;
};

#ifdef PRECISION_FLOAT
template class RodCollision<DataType3f>;
#else
template class RodCollision<DataType3d>;
#endif

}
