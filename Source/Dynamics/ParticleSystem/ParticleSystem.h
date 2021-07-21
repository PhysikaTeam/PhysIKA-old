/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of ParticleSystem class, base class of all particle-based methods
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-20
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once
#include "Framework/Framework/Node.h"

namespace PhysIKA {
template <typename TDataType>
class PointSet;

/**
 * ParticleSystem
 * Base class of all particle-based methods
 * The class is generally NOT used to instantiate a scene node, it's subclasses are.
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleSystem : public Node
{
    DECLARE_CLASS_1(ParticleSystem, TDataType)
public:
    bool self_update = true;  //!< whether the node is responsible for its update
                              //!< in some cases (quite rare), the state update of the node is handled by another node (e.g., its father)
                              //!< the class implementers should enclose the update procedure in an if statement while implementing adanvce()

    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleSystem(std::string name = "default");
    virtual ~ParticleSystem();

    /**
     * initialize cuboid-shaped particle distribution
     * Behavior is undefined if invalid arguments are specified
     *
     * @param[in] lo          lower-corner coordinate of the cuboid
     * @param[in] hi          higher-corner coordinate of the cuboid,
     *                        element values should be larger than that of lo
     * @param[in] distance    distance between particles, must be positive
     */
    void loadParticles(Coord lo, Coord hi, Real distance);

    /**
     * initialize sphere-shaped particle distribution
     * Behavior is undefined if invalid arguments are specified
     *
     * @param[in] center      coordinate of sphere center
     * @param[in] r           radius of sphere, must be positive
     * @param[in] distance    distance between particles, must be positive
     *
     */
    void loadParticles(Coord center, Real r, Real distance);

    /**
     * initialize particles from obj file
     *
     * @param[in] filename    path to the obj file
     */
    void loadParticles(std::string filename);

    /**
     * translate the particle initial configuration by a vector
     *
     * @param[in] t   the translation vector
     *
     * @return        true if succeed, false otherwise
     */
    virtual bool translate(Coord t);

    /**
     * scale the particle initial configuration
     *
     * @param[in] s   the scale factor, must be positive
     *
     * @return        true if succeed, false otherwise
     */
    virtual bool scale(Real s);

    /**
     * set current configuration as the new initial configuration
     */
    void updateTopology() override;

    /**
     * reset current configuration to initial configuration
     *
     * @return       true if succeed, false otherwise
     */
    bool resetStatus() override;

    /**
     * initialize the node
     *
     * @return       initialization status, currently always return true
     */
    bool initialize() override;

public:
    //DEF_EMPTY_CURRENT_ARRAY macro expands to a member variable definition and a getter member function
    DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");    //!< current particle positions
    DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");    //!< current particle velocities
    DEF_EMPTY_CURRENT_ARRAY(Force, Coord, DeviceType::GPU, "Force on each particle");  //!< current forces on particles

protected:
    std::shared_ptr<PointSet<TDataType>> m_pSet;  //!< point set that stores initial configuration
};

#ifdef PRECISION_FLOAT
template class ParticleSystem<DataType3f>;
#else
template class ParticleSystem<DataType3d>;
#endif
}  // namespace PhysIKA