
/**
 * @author     : Chang Yue (changyue@buaa.edu.cn)
 * @date       : 2020-08-27
 * @description: Declaration of ParticleEmitterRound class, which emits particles from a circle
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "ParticleEmitter.h"

namespace PhysIKA {

/**
 * ParticleEmitterRound, generate particles dynamically from a circle
 *
 * Usage:
 * Define a particle emitter instance, and call advance2() during simulation update.
 * The position and direction of the emitter can be adjusted by setting manipulating the emitter node.
 *
 */
template <typename TDataType>
class ParticleEmitterRound : public ParticleEmitter<TDataType>
{
    DECLARE_CLASS_1(ParticleEmitterRound, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleEmitterRound(std::string name = "particleEmitter");
    virtual ~ParticleEmitterRound();

    /**
     * particle emitting rules, it is called inside advance2()
     * users do not need to explicitly call it
     *
     * Emit particles from a circle centered at the node center, and along node direction
     * Some randomness is added to the particle distribution
     */
    void generateParticles() override;

public:
    DEF_VAR(Radius, Real, 0.05, "Emitter radius");  //!< radius of the emittign region
};

#ifdef PRECISION_FLOAT
template class ParticleEmitterRound<DataType3f>;
#else
template class ParticleEmitterRound<DataType3d>;
#endif
}  // namespace PhysIKA