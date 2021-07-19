/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Declaration of ParticleCloth class, a scene node to simulate cloth with peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-14
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {
template <typename TDataType>
class Peridynamics;

/**
 * ParticleCloth, a scene node to simulate cloth with peridynamics
 * a SurfaceMeshRender is attached to the scene node for rendering
 */
template <typename TDataType>
class ParticleCloth : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(ParticleCloth, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ParticleCloth(std::string name = "default");
    virtual ~ParticleCloth();

    void advance(Real dt) override;

    void updateTopology() override;

    bool translate(Coord t) override;
    bool scale(Real s) override;

    bool initialize() override;

    /**
     * load the surface mesh for rendering
     *
     * @param[in]    filename   path to the obj file
     */
    void loadSurface(std::string filename);

private:
    std::shared_ptr<Node>                    m_surfaceNode;  //!< node to attach SurfaceMeshRender
    std::shared_ptr<Peridynamics<TDataType>> m_peri;         //!< projective peridynamics numerical model
};

#ifdef PRECISION_FLOAT
template class ParticleCloth<DataType3f>;
#else
template class ParticleCloth<DataType3d>;
#endif
}  // namespace PhysIKA