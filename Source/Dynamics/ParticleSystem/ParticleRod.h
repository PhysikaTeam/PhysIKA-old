/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-12-22
 * @description: Declaration of ParticleRod class, projective-peridynamics based elastic rod
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include <vector>

#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "ParticleSystem.h"

namespace PhysIKA {

template <typename>
class OneDimElasticityModule;
template <typename>
class ParticleIntegrator;
template <typename>
class FixedPoints;
template <typename>
class SimpleDamping;

/**
 * ParticleRod
 * a scene node to simulate elastic rods with the approach introduced in the paper
 * <Projective Peridynamics for Modeling Versatile Elastoplastic Materials>
 *
 * The particles can be setup via loadParticles && setParticles
 * Currently masses of the particles CANNOT be set, the setMass() API of Node class does not work
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleRod : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleRod, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleRod(std::string name = "default");
    virtual ~ParticleRod();

    /**
     * Initialize the node and cooresponding modules
     *
     * @return    currently always return true
     */
    bool initialize() override;

    /**
     * Reset configuration to initial configuration
     *
     * @return    currently always return true
     */
    bool resetStatus() override;

    /**
     * advance the scene node in time
     *
     * @param[in] dt    the time interval between the states before&&after the call (deprecated)
     */
    void advance(Real dt) override;

    /**
     * setup the particles
     *
     * @param[in] particles    particle positions in order
     */
    void setParticles(std::vector<Coord> particles);

    /**
     * set the stiffness of the rod
     *
     * @param[in] stiffness     stiffness value, must be positive
     */
    void setMaterialStiffness(Real stiffness);

    /**
     * specify a particle pinned in space
     *
     * @param[in] id    id of the pinned particle, must be in range of existing particles
     * @param[in] pos   the pinned position of the particle
     */
    void addFixedParticle(int id, Coord pos);

    /**
     * remove a pinned particle
     * The particle is not removed from simulation, just not being pinned anymore
     *
     * @param[in] id    id of the pinned particle, must be in range of existing particles
     */
    void removeFixedParticle(int id);

    /**
     * get positions of the particles in host memory
     *
     * @param[out] pos  the vector that stores positions of the particles
     */
    void getHostPosition(std::vector<Coord>& pos);

    /**
     * disable all pinned particles
     */
    void removeAllFixedPositions();

    /**
     * collision handling with a plane
     *
     * @param[in] pos    position of a point on plane
     * @param[in] dir    direction of the plane
     */
    void doCollision(Coord pos, Coord dir);

    /**
     * set damping coefficient
     *
     * @param[in] d    the damping coefficient, must be positive
     */
    void setDamping(Real d);

public:
    VarField<Real> m_horizon;    //!< horizon of peridynamics approach
    VarField<Real> m_stiffness;  //!< stiffness of the rod

protected:
    DeviceArrayField<Real> m_mass;  //!< masses of each discretized particle

private:
    /**
     * reset the particle masses according to current setup
     * masses of free particles are set to 1, fixed particles are set to 1000000
     */
    void resetMassField();

private:
    std::vector<int>                                   m_fixedIds;            //!< ids of the fixed particles
    bool                                               m_modified = false;    //!< whether mass field needs reset
    std::shared_ptr<ParticleIntegrator<TDataType>>     m_integrator;          //!< integrator
    std::shared_ptr<OneDimElasticityModule<TDataType>> m_one_dim_elasticity;  //!< elastic constitutive model
    std::shared_ptr<FixedPoints<TDataType>>            m_fixed;               //!< fix point constraint
    std::shared_ptr<SimpleDamping<TDataType>>          m_damping;             //!< damping constraint
};

#ifdef PRECISION_FLOAT
template class ParticleRod<DataType3f>;
#else
template class ParticleRod<DataType3d>;
#endif
}  // namespace PhysIKA