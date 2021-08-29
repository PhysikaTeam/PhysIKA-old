#pragma once

#ifndef DEMO_SANDRIGIDCOMMON_H
#define DEMO_SANDRIGIDCOMMON_H

#include <vector>
#include <assert.h>

#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Core/Vector/vector_3d.h"

#include "Framework/Framework/Node.h"

//#include "Dynamics/Sand/SandSimulator.h"
//#include "Framework/Framework/SceneGraph.h"
//#include "Rendering/PointRenderModule.h"
//
using namespace PhysIKA;

template <typename T>
void fillGrid2D(T* grid, int nx, int ny, T value)
{
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            grid[j * nx + i] = value;
        }
    }
}

template <typename T>
void fillGrid2D(T* grid, int nx, int ny, const std::vector<int>& block, T value)
{
    assert(block.size() >= 4);

    for (int i = block[0]; i < block[1]; ++i)
    {
        for (int j = block[2]; j < block[3]; ++j)
        {
            grid[j * nx + i] = value;
        }
    }
}

template <typename T>
void fillGrid2D(PhysIKA::HeightFieldGrid<T, T, DeviceType::CPU>& grid, const std::vector<int>& block, T value)
{
    assert(block.size() >= 4);

    for (int i = block[0]; i < block[1]; ++i)
    {
        for (int j = block[2]; j < block[3]; ++j)
        {
            //grid[i*ny + j] = value;
            grid(i, j) = value;
        }
    }
}

template <typename T>
void fillGrid2D(PhysIKA::HeightFieldGrid<T, T, DeviceType::CPU>& grid, T value)
{
    int nx = grid.Nx(), ny = grid.Ny();
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            grid(i, j) = value;
        }
    }
}

bool computeBoundingBox(PhysIKA::Vector3f& center, PhysIKA::Vector3f& boxsize, const std::vector<PhysIKA::Vector3f>& vertices);

void PkAddBoundaryRigid(std::shared_ptr<Node> root, Vector3f origin, float sizex, float sizez, float boundarysize, float boundaryheight);

#endif  // DEMO_SANDRIGIDCOMMON_H