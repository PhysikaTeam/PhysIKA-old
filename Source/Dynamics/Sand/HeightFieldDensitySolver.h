#pragma once
#ifndef HEIGHTFIELDDENSITYSOLVER_H
#define HEIGHTFIELDDENSITYSOLVER_H

#include "Core/Platform.h"
#include "Core/Vector/vector_3d.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"

namespace PhysIKA {
class HeightFieldDensitySolver
{
public:
    void initialize();

    void compute(Real dt);

public:
    DeviceDArray<Vector3d>* m_gridVel = 0;

    DeviceHeightField1d* m_sandHeight = 0;
    DeviceHeightField1d* m_landHeight = 0;
};
}  // namespace PhysIKA

#endif  //HEIGHTFIELDDENSITYSOLVER_H