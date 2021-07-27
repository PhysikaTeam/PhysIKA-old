/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2021-06-17
 * @description: Declaration of ParticleFluidFast class
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-21
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/ParticleEmitter.h"

namespace PhysIKA {

/**
 * ParticleFluidFast
 * a scene node for particle-based fluid methods
 * The default solver is PBD
 * reference: Macklin and Muller's "Position Based Fluids"
 * Solver can be specified by calling setNumericalModel()
 * Spatial hashing is used to reorder the particles for fast neighbor search and less memory load
 *
 * Known issues(TODO: Fix them!):
 * 1. particle emitters are not used in implementation
 * 2. a hard-coded range is used in spatial hashing
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleFluidFast : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleFluid, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleFluidFast(std::string name = "default");
    virtual ~ParticleFluidFast();

    /**
     * advance the scene node in time
     *
     * @param[in] dt    the time interval between the states before&&after the call (deprecated)
     */
    void advance(Real dt) override;
    /**
     * clear particles in scene if there's any particle emitters,
     * otherwise reset particles' state to initial configuration
     *
     * @return     true if succeed, false otherwise
     */
    bool resetStatus() override;

public:
    DEF_EMPTY_CURRENT_ARRAY(PositionInOrder, Coord, DeviceType::GPU, "Particle position");    //!< temp array for reordered particle positions
    DEF_EMPTY_CURRENT_ARRAY(VelocityInOrder, Coord, DeviceType::GPU, "Particle velocity");    //!< temp array for reordered particle velocities
    DEF_EMPTY_CURRENT_ARRAY(ForceInOrder, Coord, DeviceType::GPU, "Force on each particle");  //!< temp array for reordered particle forces

    DeviceArray<int> ids;         //!<spatial grid ids corresponding to particles
    DeviceArray<int> idsInOrder;  //!<particle ids

private:
    DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter<TDataType>, "Particle Emitters");  //!< particle emitters and corresponding accessors
};

#ifdef PRECISION_FLOAT
template class ParticleFluidFast<DataType3f>;
#else
template class ParticleFluidFast<DataType3d>;
#endif
}  // namespace PhysIKA