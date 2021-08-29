/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-25
 * @description: Declaration of ParticleElastoplasticBody class, projective-peridynamics based elastoplastic bodies
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-21
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once
#include "ParticleSystem.h"

namespace PhysIKA {
template <typename>
class NeighborQuery;
template <typename>
class ParticleIntegrator;
template <typename>
class ElastoplasticityModule;
template <typename TDataType>
class ImplicitViscosity;

/**
 * ParticleElastoplasticBody
 * a scene node to simulate elastoplastic bodies with the approach introduced in the paper
 * <Projective Peridynamics for Modeling Versatile Elastoplastic Materials>
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleElastoplasticBody : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleElastoplasticBody, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleElastoplasticBody(std::string name = "default");
    virtual ~ParticleElastoplasticBody();

    /**
     * advance the scene node in time
     *
     * @param[in] dt    the time interval between the states before&&after the call (deprecated)
     */
    void advance(Real dt) override;

    /**
     * set current configuration as the new initial configuration
     * the surface node is updated as well
     */
    void updateTopology() override;

    /**
     * initialize the node
     * @issue: Did the constructor do too much?
     *
     * @return       initialization status, currently always return true
     */
    bool initialize() override;

    /**
     * translate the particle initial configuration by a vector
     * the surface node is updated as well
     * Hence if this function is called during simulation, it will cause inconsistency between
     * the node's current particle configuration and surface mesh.
     *
     * @param[in] t   the translation vector
     *
     * @return        true if succeed, false otherwise
     */
    bool translate(Coord t) override;

    /**
     * scale the particle initial configuration
     * the surface node is updated as well
     * Hence if this function is called during simulation, it will cause inconsistency between
     * the node's current particle configuration and surface mesh.
     *
     * @param[in] s   the scale factor, must be positive
     *
     * @return        true if succeed, false otherwise
     */
    bool scale(Real s) override;

    /**
     * load surface mesh from obj file
     *
     * @param[in] filename    path to the obj file
     */
    void loadSurface(std::string filename);

    /**
     * setter of the elastoplasticity module
     */
    void setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver);

    /**
     * return the node representing the surface
     */
    std::shared_ptr<Node> getSurfaceNode()
    {
        return m_surfaceNode;
    }

public:
    VarField<Real> m_horizon;

private:
    std::shared_ptr<Node> m_surfaceNode;  //!< surface mesh node, generally for rendering purposes

    std::shared_ptr<ParticleIntegrator<TDataType>>     m_integrator;  //!< the integrator to step forward in time
    std::shared_ptr<NeighborQuery<TDataType>>          m_nbrQuery;    //!< query particle neighbors
    std::shared_ptr<ElastoplasticityModule<TDataType>> m_plasticity;  //!< elastoplasticity constitutive model
    std::shared_ptr<ImplicitViscosity<TDataType>>      m_visModule;   //!< viscosity model
};

#ifdef PRECISION_FLOAT
template class ParticleElastoplasticBody<DataType3f>;
#else
template class ParticleElastoplasticBody<DataType3d>;
#endif
}  // namespace PhysIKA