/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of ParticleIntegrator class, used to update velocity and position at each time step
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-05
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/NumericalIntegrator.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"

/**
 * ParticleIntegrator, update velocity and position at each time step using gravity and force desity
 * 
 * Usage:
 * 1. Initialize Position, Velocity and ForceDensity
 * 2. Call integrate in each step
 *
 */

namespace PhysIKA {
template <typename TDataType>
class ParticleIntegrator : public NumericalIntegrator
{
    DECLARE_CLASS_1(ParticleIntegrator, TDataType)

public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleIntegrator();
    ~ParticleIntegrator() override{};

    void begin() override;  //seems to have no influence on current version
    void end() override;    //seems to have no influence on current version

    /**
     * do particle integration
     * 
     * update velocity and positions for all particles, so inPosition and inVelocity should be initialized before calling
     *
     * @return true(always)
     */
    bool integrate() override;

    /**
     * Called to update velocities of particles
     * 
     * Add force density and gravity onto velocities of particles
     * @return true(always)
     */
    bool updateVelocity();

    /**
     * Called to update positions of particles
     * 
     * Use the new velocities to update positions
     * @return true(always)
     */
    bool updatePosition();

protected:
    bool initializeImpl() override;  //seems to have no influence on current version

public:
    /**
        * @brief Position
        * Particle position
        */
    DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

    /**
        * @brief Velocity
        * Particle velocity
        */
    DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

    /**
        * @brief Force density
        * Force density on each particle
        */
    DEF_EMPTY_IN_ARRAY(ForceDensity, Coord, DeviceType::GPU, "Force density on each particle");

private:
    DeviceArray<Coord> m_prePosition;  //seems to have no influence on current version
    DeviceArray<Coord> m_preVelocity;  //seems to have no influence on current version
};

#ifdef PRECISION_FLOAT
template class ParticleIntegrator<DataType3f>;
#else
template class ParticleIntegrator<DataType3d>;
#endif
}  // namespace PhysIKA