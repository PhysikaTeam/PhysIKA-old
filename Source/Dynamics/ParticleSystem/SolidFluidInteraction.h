/**
 * @author     : He Xiaowei (xiaowei@iscas.ac.cn)
 * @date       : 2020-10-07
 * @description: Declaration of SolidFluidInteraction class, applies solid-fluid interaction by PBD
 * @version    : 1.0
 * 
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: poslish code
 * @version    : 1.1
 * 
 */
#pragma once
#include "Framework/Framework/Node.h"

namespace PhysIKA {
template <typename T>
class RigidBody;
template <typename T>
class ParticleSystem;
template <typename T>
class NeighborQuery;
template <typename T>
class DensityPBD;

/**
 * SolidFluidInteraction, coulping solid and fluid using position based dynamics
 * Usage:
 * 1. Define a SolidFluidInteraction instance
 * 2. Initialize by calling addParticleSystem
 * 3. Call advance in each loop
 */

template <typename TDataType>
class SolidFluidInteraction : public Node
{
    DECLARE_CLASS_1(SolidFluidInteraction, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SolidFluidInteraction(std::string name = "SolidFluidInteration");
    ~SolidFluidInteraction() override;

public:
    /*Returns true*/
    bool initialize() override;

    /*Currently not supported*/
    bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);

    /**
     * add particle system to particle system list
     *
     * @param[in] child    the particle system to be added
     */
    bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

    /*Initialize all intermediate arrays*/
    bool resetStatus() override;

    /**
     * solve the solid fluid interaction
     *
     * @param[in] dt    time step
     */
    void advance(Real dt) override;

    /**
     * set smoothingLength
     *
     * @param[in] d    set radius to be d and sampling distance to be d/2, may lead to problems
     */
    void setInteractionDistance(Real d);

private:
    VarField<Real> radius;

    DeviceArrayField<Coord> m_position;
    DeviceArrayField<Real>  m_mass;
    DeviceArrayField<Coord> m_vels;

    DeviceArray<int> m_objId;

    DeviceArray<Coord> posBuf;
    DeviceArray<Real>  weights;
    DeviceArray<Coord> init_pos;

    std::shared_ptr<NeighborList<int>>        m_nList;
    std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;

    std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
    ;

    std::vector<std::shared_ptr<RigidBody<TDataType>>>      m_rigids;
    std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
};

#ifdef PRECISION_FLOAT
template class SolidFluidInteraction<DataType3f>;
#else
template class SolidFluidInteraction<DataType3d>;
#endif
}  // namespace PhysIKA