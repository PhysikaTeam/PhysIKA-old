/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Declaration of MeshCollision class, solving the non-intersection constraint between particles and triangle mesh
 * @version    : 1.1
 */

#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/CollisionModel.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/FieldArray.h"

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
class PointSet;
template <typename TDataType>
class TriangleSet;

/**
 * MeshCollision, a module handels the collision between particles and triangle meshes
 * Positions and velocities of particles are updated
 * 
 * Usage:
 * 1. Define a MeshCollision instance
 * 2. Initialize by connecting m_position, m_velocity, m_triangle_vertex, m_triangle_index, m_neighborhood_tri
 * 3. Call initialize() to set up sizes of related device arrays
 * 4. Call doCollision in simulation loop
 *
 * 
 * Currently used in PositionBasedFluidModelMesh and SemiAnalyticalIncompressibleFluidModel
 */

template <typename TDataType>
class MeshCollision : public CollisionModel
{
    DECLARE_CLASS_1(MeshCollision, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    MeshCollision();
    virtual ~MeshCollision();

    /**
     * Check if the topology is supported
     * The design of is support seems to be deprecated.
     *
     * @return    true if the collidable object is point, false otherwise
     */
    bool isSupport(std::shared_ptr<CollidableObject> obj) override;

    /**
     * Add a Collidable Object
     * The design of addCollidableObject is deprecated.
     *
     * @push back obj into m_collidableObjects
     */
    void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

    /**
     * Resize all intermediate device arrays 
     * 
     * m_position, m_velocity, m_triangle_index, m_triangle_vertex and m_neighborhood_tri has to be initialized to ensure correctness
     * 
     * @return true(always)
     */
    bool initializeImpl() override;

    /**
     * Solve the collisions between triangles and particles
     * 
     * m_position, m_velocity, m_triangle_index, m_triangle_vertex and m_neighborhood_tri has to be initialized to ensure correctness
     * 
     */
    void doCollision() override;

    DeviceArrayField<Coord>    m_position;              //input and output, current positions of particles
    DeviceArrayField<Coord>    m_velocity;              //input and output, current velocities of particles
    DeviceArrayField<Real>     m_triangle_vertex_mass;  //input, mass of triangles, currently no use, reserved for future
    DeviceArrayField<Coord>    m_triangle_vertex;       //positions of triangle vertexs
    DeviceArrayField<Coord>    m_triangle_vertex_old;   //positions of triangle vertexs at last time step, reserved for CCD
    DeviceArrayField<Triangle> m_triangle_index;        //IDs of triangles
    DeviceArrayField<int>      m_flip;                  //to check if the norm of each triangle is flipped
    NeighborField<int>         m_neighborhood_tri;      //neighbor list of particle-triangle, note that only [particle-triangle] pairs are stored

    DeviceArrayField<Coord> m_velocity_mod;  //norm of velocity of each particle

protected:
    DeviceArray<int> m_objId;

    DeviceArray<Real>  weights;
    DeviceArray<Coord> init_pos;
    DeviceArray<Coord> posBuf;

    DeviceArray<Coord> m_position_previous;
    DeviceArray<Coord> m_triangle_vertex_previous;

    std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
    std::shared_ptr<NeighborList<int>>        m_nList;

    std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;
};

#ifdef PRECISION_FLOAT
template class MeshCollision<DataType3f>;
#else
template class MeshCollision<DataType3d>;
#endif

}  // namespace PhysIKA
