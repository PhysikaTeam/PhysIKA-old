/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of StaticBoundary class, representing static objects in scene that can couple with simulated objects
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {

template <typename TDataType>
class BoundaryConstraint;

/**
 * StaticBoundary, the static objects in scene
 * Currently coupling with rigidbody and particlesystem are supported
 * Issue(TODO): rigid body is not handled yet.
 *
 * Usage:
 * 1. Define a StaticBoundary instance
 * 2. Initialize one or more objects by loading from SDF or analytical representation
 * 3. Register the simulated objects by calling addRigidBody/addParticleSystem
 * 4. call advance() in simulation loop
 *
 * Note on visualize the static boundary:
 * add a rigid body node with same position/direction in scene,
 * set the rigid body as inactive, and attach a render module to the rigid body
 */
template <typename TDataType>
class StaticBoundary : public Node
{
    DECLARE_CLASS_1(StaticBoundary, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    StaticBoundary();
    ~StaticBoundary() override;

    /*
     * perform one-way coupling between static boundary and registered simulated objects
     *
     * @param[in] dt    time interval between the state change
     */
    void advance(Real dt) override;

    /**
     * add a static object into the scene by loading from signed distance file
     *
     * @param[in] filename        file name of the sdf file
     * @param[in] bOutBoundary    the boundary acts as a container, if bOutBoundary is set true
     *                            otherwise it is a normal static object
     */
    void loadSDF(std::string filename, bool bOutBoundary = false);

    /**
     * add a cuboid-shaped static boundary
     *
     * @param[in] lo              coordinate of the cuboid's lower corner
     * @param[in] hi              coordinate of the cuboid's higher corner
     * @param[in] distance        sampling distance of the signed distance for the cuboid
     * @param[in] bOutBoundarythe boundary acts as a container, if bOutBoundary is set true
     *                            otherwise it is a normal static object
     */
    void loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);

    /**
     * add a sphere-shaped static boundary
     *
     * @param[in] center          center of the sphere
     * @param[in] distance        sampling distance of the signed distance for the sphere
     * @param[in] bOutBoundarythe boundary acts as a container, if bOutBoundary is set true
     *                            otherwise it is a normal static object
     */
    void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false);

    /**
     * translate the static objects registered to the static boundary instance
     *
     * @param[in] t    the translation vector
     */
    void translate(Coord t);

    /**
     * scale the static objects registered to the static boundary instance
     *
     * @param[in] s    the scale factor, must be positive
     */
    void scale(Real s);

private:
    std::vector<std::shared_ptr<BoundaryConstraint<TDataType>>> m_obstacles;  //!< each static object is registered as one boundary constraint

    DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");                //!< coupling rigidbodies and corresponding accessors
    DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");  //!< coupling particle systems and corresponding accessors
};

#ifdef PRECISION_FLOAT
template class StaticBoundary<DataType3f>;
#else
template class StaticBoundary<DataType3d>;
#endif

}  // namespace PhysIKA
