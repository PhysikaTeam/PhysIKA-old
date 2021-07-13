#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

#include <map>

namespace PhysIKA {

template <typename TDataType>
class SimpleDamping : public ConstraintModule
{
    DECLARE_CLASS_1(SimpleDamping, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SimpleDamping();
    ~SimpleDamping() override;

    bool constrain() override;

    void setDampingCofficient(Real c);

public:
    /**
        * @brief Particle velocity
        */
    DeviceArrayField<Coord> m_velocity;

protected:
    virtual bool initializeImpl() override;

private:
    VarField<float> m_damping;
};

#ifdef PRECISION_FLOAT
template class SimpleDamping<DataType3f>;
#else
template class SimpleDamping<DataType3d>;
#endif

}  // namespace PhysIKA
