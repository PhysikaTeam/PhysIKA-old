
/**
 * @author     : Chang Yue (changyue@buaa.edu.cn)
 * @date       : 2020-08-27
 * @description: Declaration of ParticleEmitterSquare class, which emits particles from a square
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
 * ParticleEmitterSquare, generate particles dynamically from a square
 *
 * Usage:
 * Define a particle emitter instance, and call advance2() during simulation update.
 * The position and direction of the emitter can be adjusted by setting manipulating the emitter node.
 *
 */
template <typename TDataType>
class ParticleEmitterSquare : public ParticleEmitter<TDataType>
{
    DECLARE_CLASS_1(ParticleEmitterSquare, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleEmitterSquare(std::string name = "particleEmitter");
    virtual ~ParticleEmitterSquare();

    /**
     * particle emitting rules, it is called inside advance2()
     * users do not need to explicitly call it
     *
     * Emit particles from a square centered at the node center, and along node direction
     * Some randomness is added to the particle distribution
     */
    void generateParticles() override;

private:
    DEF_VAR(Width, Real, 0.05, "Emitter width");    //!< width of the emitting region
    DEF_VAR(Height, Real, 0.05, "Emitter height");  //!< height of the emitting region
};

#ifdef PRECISION_FLOAT
template class ParticleEmitterSquare<DataType3f>;
#else
template class ParticleEmitterSquare<DataType3d>;
#endif
}  // namespace PhysIKA