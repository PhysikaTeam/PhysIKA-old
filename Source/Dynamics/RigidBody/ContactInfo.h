#pragma once

#include "Core/Vector/vector_3d.h"

namespace PhysIKA {
template <typename TReal>
struct ContactInfo
{

public:
    int              id0 = 0, id1 = 0;
    TReal            mu = 1.0;
    Vector<TReal, 3> point0;
    Vector<TReal, 3> point1;
    Vector<TReal, 3> normal;
};

template <typename TReal>
struct PointTriContact
{
public:
    int              id0 = 0;
    int              id1 = 0;
    float            mu  = 1.0;
    Vector<TReal, 3> point0;

    // Triangle points.
    Vector<TReal, 3> triP0;
    Vector<TReal, 3> triP1;
    Vector<TReal, 3> triP2;
};
}  // namespace PhysIKA