/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Declaration of ParticleViscoplasticBody, simulate viscoplasticity with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {
template <typename>
class NeighborQuery;
template <typename>
class PointSetToPointSet;
template <typename>
class ParticleIntegrator;
template <typename>
class ElasticityModule;
template <typename>
class ElastoplasticityModule;
template <typename TDataType>
class ImplicitViscosity;

/**
 * ParticleViscoplasticBody, a scene node to simulate viscoplasticity with projective-peridynamics
 * a SurfaceMeshRenderer is attached to the scene node for rendering
 */
template <typename TDataType>
class ParticleViscoplasticBody : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleViscoplasticBody, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleViscoplasticBody(std::string name = "default");
    virtual ~ParticleViscoplasticBody();

    void advance(Real dt) override;

    void updateTopology() override;

    bool initialize() override;

    bool translate(Coord t) override;
    bool scale(Real s) override;

    /**
     * load the surface mesh for rendering
     *
     * @param[in]    filename   path to the obj file
     */
    void loadSurface(std::string filename);

public:
    VarField<Real> m_horizon;

private:
    std::shared_ptr<Node> m_surfaceNode;

    std::shared_ptr<ParticleIntegrator<TDataType>>     m_integrator;
    std::shared_ptr<NeighborQuery<TDataType>>          m_nbrQuery;
    std::shared_ptr<ElasticityModule<TDataType>>       m_elasticity;
    std::shared_ptr<ElastoplasticityModule<TDataType>> m_plasticity;
    std::shared_ptr<ImplicitViscosity<TDataType>>      m_visModule;
};

#ifdef PRECISION_FLOAT
template class ParticleViscoplasticBody<DataType3f>;
#else
template class ParticleViscoplasticBody<DataType3d>;
#endif
}  // namespace PhysIKA