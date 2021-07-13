#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {
template <typename TDataType>
class ImplicitViscosity : public ConstraintModule
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ImplicitViscosity();
    ~ImplicitViscosity() override;

    bool constrain() override;

    void setIterationNumber(int n);

    void setViscosity(Real mu);

protected:
    bool initializeImpl() override;

public:
    VarField<Real> m_viscosity;
    VarField<Real> m_smoothingLength;

    DeviceArrayField<Coord> m_velocity;
    DeviceArrayField<Coord> m_position;

    NeighborField<int> m_neighborhood;

private:
    int m_maxInteration;

    DeviceArray<Coord> m_velOld;
    DeviceArray<Coord> m_velBuf;
};

#ifdef PRECISION_FLOAT
template class ImplicitViscosity<DataType3f>;
#else
template class ImplicitViscosity<DataType3d>;
#endif
}  // namespace PhysIKA