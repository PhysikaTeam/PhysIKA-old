/**
 * @author     : He Xiaowei (xiaowei@iscas.ac.cn)
 * @date       : 2020-10-07
 * @description: Declaration of RodCollision class, applies point-wise collision handeling for rods
 * @version    : 1.0
 * 
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: poslish code
 * @version    : 1.1
 * 
 */
#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/CollisionModel.h"
/**
 * RodCollision
 * a CollisionModel for particles
 *
 * Usage:
 * 1. Create a RodCollision model in the parent node
 * 2. Initialize by calling addCollidableObject
 * 3. Call doCollision in each loop
 *
 */
namespace PhysIKA {
template <typename>
class CollidablePoints;
template <typename>
class NeighborQuery;
template <typename>
class NeighborList;
template <typename>
class GridHash;

template <typename TDataType>
class RodCollision : public CollisionModel
{
    DECLARE_CLASS_1(RodCollision, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    RodCollision();
    virtual ~RodCollision();

    /**
     * Check if a given collidable object is supported, only points are supported by this class
     *
     * @param[in]      obj          the collidable object of interest
     *
     * @return true if obj is point
     */
    bool isSupport(std::shared_ptr<CollidableObject> obj) override;

    /**
     * Add collidable object into m_collidableObjects. When added, collision handeling is applied to the object.
     * Used to initialize the class
     *
     * @param[in]      obj          the collidable object of interest
     *
     * @do nothing if obj is not supported, add to m_collidableObjects otherwise
     */
    void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

    /**
     * Initialize the node and cooresponding modules
     *
     * @return    true if all fields are ready, false otherwise
     */
    bool initializeImpl() override;

    /**
    * Can be called to enforce collision, update the velocities and positions of particles
    * m_objId&&m_points&&m_vels should be initialized before calling this API
    */
    void doCollision() override;

protected:
    DeviceArray<int>   m_objId;   //object IDs, indicating the object each point belongs
    DeviceArray<Coord> m_points;  //particle positions
    DeviceArray<Coord> m_vels;    //particle velocities

    std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;  //!< node to calculate neighborhood
    std::shared_ptr<NeighborList<int>>        m_nList;     //!< neighbor list of particles

    std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;  //!< all objects to perform collisions
};

#ifdef PRECISION_FLOAT
template class RodCollision<DataType3f>;
#else
template class RodCollision<DataType3d>;
#endif

}  // namespace PhysIKA
