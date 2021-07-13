#pragma once
#include <vector>
#include "Core/Array/Array.h"
#include "Framework/Topology/DistanceField3D.h"
#include "Framework/Framework/CollidableObject.h"

namespace PhysIKA {

template <typename TDataType>
class CollidableSDF : public CollidableObject
{
    DECLARE_CLASS_1(CollidableSDF, TDataType)
public:
    typedef typename TDataType::Real   Real;
    typedef typename TDataType::Coord  Coord;
    typedef typename TDataType::Matrix Matrix;

    CollidableSDF();

    void setSDF(std::shared_ptr<DistanceField3D<TDataType>> sdf)
    {
        if (m_sdf != nullptr)
        {
            m_sdf->release();
        }

        m_sdf = sdf;
    }

    std::shared_ptr<DistanceField3D<TDataType>> getSDF()
    {
        return m_sdf;
    }

    ~CollidableSDF()
    {
        m_sdf->release();
    }

    void updateCollidableObject() override{};
    void updateMechanicalState() override{};

    Coord getTranslation()
    {
        return m_translation;
    }
    Matrix getRotationMatrix()
    {
        return m_rotation;
    }

    bool initializeImpl() override;

private:
    Coord  m_translation;
    Matrix m_rotation;

    Coord m_velocity;
    Coord m_angular_velocity;

    std::shared_ptr<DistanceField3D<TDataType>> m_sdf;
};

#ifdef PRECISION_FLOAT
template class CollidableSDF<DataType3f>;
#else
template class CollidableSDF<DataType3d>;
#endif
}  // namespace PhysIKA