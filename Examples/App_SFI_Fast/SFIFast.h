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

/*!
          *        class    SolidFluidInteraction
          *        brief    Position-based fluids.
       *
          *    This class implements a position-based fluid solver.
          *    Refer to Macklin and Muller's "Position Based Fluids" for details
       *
       */

template <typename TDataType>
class SFIFast : public Node
{
    DECLARE_CLASS_1(SolidFluidInteraction, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SFIFast(std::string name = "SolidFluidInteration");
    ~SFIFast() override;

public:
    bool initialize() override;

    bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
    bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

    bool resetStatus() override;

    void advance(Real dt) override;

    void setInteractionDistance(Real d);

private:
    VarField<Real> radius;

    DeviceArrayField<Coord> m_position;
    DeviceArrayField<Coord> m_vels;
    DeviceArrayField<Coord> m_force;
    DeviceArrayField<Real>  m_mass;

    DeviceArray<int> m_objId;

    DeviceArray<Coord> posBuf;
    DeviceArray<Real>  weights;
    DeviceArray<Coord> init_pos;

    std::shared_ptr<NeighborList<int>> m_nList;

    std::vector<std::shared_ptr<RigidBody<TDataType>>>      m_rigids;
    std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
};

#ifdef PRECISION_FLOAT
template class SFIFast<DataType3f>;
#else
template class SFIFast<DataType3d>;
#endif
}  // namespace PhysIKA