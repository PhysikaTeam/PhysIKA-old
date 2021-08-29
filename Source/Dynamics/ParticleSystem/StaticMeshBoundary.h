/**
 * @author     : Chang Yue (changyue@buaa.edu.cn)
 * @date       : 2020-09-03
 * @description: Declaration of StaticBoundaryMesh class, representing mesh-based static objects in scene that can couple with simulated objects
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-26
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {

template <typename T>
class TriangleSet;
template <typename T>
class NeighborQuery;

/**
 * StaticMeshBoundary, the static objects represented as surface mesh in scene
 * Currently coupling with particle system is supported
 *
 * Usage:
 * 1. Define a StaticMeshBoundary instance
 * 2. Initialize one or more objects by loading meshes
 * 3. Register the simulated objects by calling addParticleSystem
 * 4. call advance() in simulation loop
 *
 * Note on visualize the static mesh boundary
 * attach a render module to the node
 */
template <typename TDataType>
class StaticMeshBoundary : public Node
{
    DECLARE_CLASS_1(StaticMeshBoundary, TDataType)
public:
    // typename =>  to tell compiler that this is a type
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    StaticMeshBoundary();
    ~StaticMeshBoundary() override;

    /**
     * add a static object into the scene by loading from an obj file
     *
     * @param[in] filename    file name of the obj file
     */
    void loadMesh(std::string filename);

    /*
     * perform one-way coupling between static boundary and registered simulated objects
     *
     * @param[in] dt    time interval between the state change
     */
    void advance(Real dt) override;

    /**
     * do nothing, always return true
     */
    bool initialize() override;

    /**
     * fill in particle position&&velcity, triangle vertex&&index entries of StaticMeshBoundary
     * setup neighbor search
     *
     * Issue: resetStatus performs more like initialize？
     */
    bool resetStatus() override;

public:
    DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");                //!< coupling rigidbodies and corresponding accessors
    DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");  //!< coupling particle systems and corresponding accessors

    DEF_VAR(ImportFile, std::string, "", "Solver");  //!< GUI stuff, added by HNU, needs poslishing

public:
    DEF_EMPTY_CURRENT_ARRAY(ParticlePosition, Coord, DeviceType::GPU, "Particle position");  //!< temp buffer to store positions of simulated objects
    DEF_EMPTY_CURRENT_ARRAY(ParticleVelocity, Coord, DeviceType::GPU, "Particle velocity");  //!< temp buffer to store velocities of simulated objects
    DEF_EMPTY_CURRENT_ARRAY(TriangleVertex, Coord, DeviceType::GPU, "Triangle Vertex");      //!< temp buffer to store static mesh vertices
    DEF_EMPTY_CURRENT_ARRAY(TriangleIndex, Triangle, DeviceType::GPU, "Triangle Index");     //!< temp buffer to store static mesh triangle indices

private:
    std::shared_ptr<NeighborQuery<TDataType>>            m_nbrQuery;   //!< module to perform neighbor search between simulated particles and mesh vertices
    VarField<Real>                                       radius;       //!< neighbor search radius
    std::vector<std::shared_ptr<TriangleSet<TDataType>>> m_obstacles;  //!< each static object is registered as a triangleSet
};

#ifdef PRECISION_FLOAT
template class StaticMeshBoundary<DataType3f>;
#else
template class StaticMeshBoundary<DataType3d>;
#endif

}  // namespace PhysIKA
