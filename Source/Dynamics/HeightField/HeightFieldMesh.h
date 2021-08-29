#pragma once
#ifndef HEIGHTFIELDMESH_H
#define HEIGHTFIELDMESH_H

#include "Framework/Topology/TriangleSet.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"

#include <memory>

namespace PhysIKA {

/**
    * @brief Height field mesh generation.
    */
class HeightFieldMesh
{

public:
    /**
        * @brief Generate height field mesh, including mesh TRIANGLES and mesh VERTICES.
        */
    void generate(std::shared_ptr<TriangleSet<DataType3f>> triset, DeviceHeightField1d& hfield);
};

}  // namespace PhysIKA

#endif  //HEIGHTFIELDMESH_H