#pragma once
#include "../math/geometry.h"

struct SortingGrid
{
    cfloat3 xmin;
    float   dx;
    cint3   res;
    uint*   cell_start;
    uint*   cell_end;
};

void computeParticleHash_host(
    cfloat3* pos,
    uint*    particle_hash,
    uint*    particle_index,
    cfloat3  xmin,
    float    dx,
    cint3    res,
    int      num_particles);

void sortParticleHash(
    uint* particle_hash,
    uint* particle_index,
    int   num_particles);

void findCellStart_host(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,
    int   num_particles,
    int   num_cells);

void FindCellStartHost(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,

    uint* sortedIndices,
    uint* indicesAfterSort,
    int   num_particles,
    int   num_cells);