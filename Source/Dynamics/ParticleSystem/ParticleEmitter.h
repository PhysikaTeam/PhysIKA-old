/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2020-08-22
 * @description: Declaration of ParticleEmitter class, base class of all particle emitters that generate particles for simulation
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "ParticleSystem.h"

namespace PhysIKA {

/**
 * ParticleEmitter, base class of all particle emitters.
 * Particle emitters dynamically genreate particles for simulation.
 *
 * Usage:
 * Define a particle emitter instance, and call advance2() during simulation update.
 *
 * For subclass developers:
 * Override generateParticles() to create new particle emitting rules.
 */
template <typename TDataType>
class ParticleEmitter : public ParticleSystem<TDataType>
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleEmitter(std::string name = "particle emitter");
    virtual ~ParticleEmitter();

    /**
     * update the particle states
     * call this function in advance() call of simulation pipeline
     *
     * @param[in] dt    time interval between state update
     */
    void advance2(Real dt);

    /**
     * particle emitting rules, it is called inside advance2()
     * Different particle emitters implement different rules
     */
    virtual void generateParticles();

    void advance(Real dt) override;  // deprecated in ParticleEmitter class hierarchy
    void updateTopology() override;  // deprecated in ParticleEmitter class hierarchy

    /**
     * Clear already-generated particles
     *
     * @return    always return true
     */
    bool resetStatus() override;

private:
    DEF_VAR(VelocityMagnitude, Real, 1, "Emitter Velocity");              //!< magnitude of emitting velocity
    DEF_VAR(SamplingDistance, Real, 0.005, "Emitter Sampling Distance");  //!< sampling distance between particles

protected:
    DeviceArray<Coord> gen_pos;  //!< particle positions generated in each generateParticles() call
    DeviceArray<Coord> gen_vel;  //!< particle velocities generated in each generateParticles() call

    DeviceArray<Coord> pos_buf;    //!< temporary position buffer
    DeviceArray<Coord> vel_buf;    //!< temporary velocity buffer
    DeviceArray<Coord> force_buf;  //!< temporary force buffer
};

#ifdef PRECISION_FLOAT
template class ParticleEmitter<DataType3f>;
#else
template class ParticleEmitter<DataType3d>;
#endif
}  // namespace PhysIKA