#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Framework/CollisionModel.h"

namespace Physika
{
template <typename> class CollidablePoints;
template <typename> class GridHash;

template<typename TDataType>
class CollisionPoints : public CollisionModel
{
	DECLARE_CLASS_1(CollisionPoints, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	CollisionPoints();
	virtual ~CollisionPoints();

	bool isSupport(std::shared_ptr<CollidableObject> obj) override;

	void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

	bool initializeImpl() override;

	void doCollision() override;
	
protected:
	DeviceArray<int> m_objId;
	DeviceArray<Coord> m_points;
	DeviceArray<Coord> m_vels;

	std::shared_ptr<GridHash<TDataType>> m_gHash;

	std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;
};

#ifdef PRECISION_FLOAT
template class CollisionPoints<DataType3f>;
#else
template class CollisionPoints<DataType3d>;
#endif

}
