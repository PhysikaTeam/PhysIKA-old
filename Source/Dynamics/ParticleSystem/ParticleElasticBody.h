/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of ParticleElasticBody class, projective-peridynamics based elastic bodies
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-20
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "ParticleSystem.h"

namespace PhysIKA {
template <typename>
class ElasticityModule;
template <typename>
class PointSetToPointSet;

/**
 * ParticleElasticBody
 * a scene node to simulate elastic bodies with the approach introduced in the paper
 * <Projective Peridynamics for Modeling Versatile Elastoplastic Materials>
 *
 * @param TDataType  template parameter that represents aggregation of scalar, vector, matrix, etc.
 */
template <typename TDataType>
class ParticleElasticBody : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleElasticBody, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleElasticBody(std::string name = "default");
    virtual ~ParticleElasticBody();

    /**
     * initialize the node
     * @issue: Did the constructor do too much?
     *
     * @return       initialization status, currently always return true
     */
    bool initialize() override;

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
     * setter and getter of the elasticity module
     */
    void                                         setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);
    std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();

    /**
     * load surface mesh from obj file
     *
     * @param[in] filename    path to the obj file
     */
    void loadSurface(std::string filename);

    /**
     * get mapping from particles to surface mesh
     */
    std::shared_ptr<PointSetToPointSet<TDataType>> getTopologyMapping();

    /**
     * return the node representing the surface
     */
    std::shared_ptr<Node> getSurfaceNode()
    {
        return m_surfaceNode;
    }

public:
    DEF_EMPTY_VAR(Horizon, Real, "Horizon");  //!< horizon variable of peridynamics
                                              //!< DEF_EMPTY_VAR macro expands to the definition of
                                              //!< a private member var_Horizon and a public function varHorizon()

private:
    std::shared_ptr<Node> m_surfaceNode;  //!< surface mesh node, generally for rendering purposes
};

#ifdef PRECISION_FLOAT
template class ParticleElasticBody<DataType3f>;
#else
template class ParticleElasticBody<DataType3d>;
#endif
}  // namespace PhysIKA