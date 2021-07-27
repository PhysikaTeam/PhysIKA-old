/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2018-12-17
 * @description: Declaration of ParticleFluid class, which is a container for particle-based fluid solvers
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-21
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace PhysIKA {
/**
 * ParticleFluid
 * a scene node for particle-based fluid methods
 * The default solver is PBD
 * reference: Macklin and Muller's "Position Based Fluids"
 * Solver can be specified by calling setNumericalModel()
 *
 * The source of fluids can be setup exclusively  in 2 ways:
 * 1. through multiple particle emitters (dynamic fluid source)
 * 2. via loadParticles() function call  (static fluid source)
 *
 * It may lead to undefined behavior if 2 ways are applied together
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleFluid : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleFluid, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleFluid(std::string name = "default");
    virtual ~ParticleFluid();

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

private:
    DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter<TDataType>, "Particle Emitters");  //!< particle emitters and corresponding accessors

    DEF_VAR(ImportFile, std::string, "", "ImportFile");  //!< Qt GUI stuff, added by HNU, need polishing
};

#ifdef PRECISION_FLOAT
template class ParticleFluid<DataType3f>;
#else
template class ParticleFluid<DataType3d>;
#endif
}  // namespace PhysIKA