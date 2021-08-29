#pragma once
#include "Core/Array/Array3D.h"

namespace PhysIKA {
template <typename TDataType>
class UniformGrid3D
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    UniformGrid3D(){};
    ~UniformGrid3D(){};

private:
    DeviceArray3D<Coord> m_coords;
};
}  // namespace PhysIKA
