#pragma once
#include "Core/DataTypes.h"
#include "Core/Utility.h"
#include "Core/Array/Array.h"
#include "Framework/Topology/NeighborList.h"
#include "Core/Utility/Scan.h"
#include "EdgeSet.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/Primitive3D.h"

namespace PhysIKA {

#define INVALID -1
#define BUCKETS 8
#define CAPACITY 16

template <typename TDataType>
class GridHash
{
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    GridHash();
    ~GridHash();

    void setSpace(Real _h, Coord _lo, Coord _hi);

    void construct(DeviceArray<Coord>& pos);
    void construct(DeviceArray<Coord>& pos, DeviceArray<Triangle>& tri, DeviceArray<Coord>& Tri_pos);

    void clear();

    void release();

    GPU_FUNC inline int getIndex(int i, int j, int k)
    {
        if (i < 0 || i >= nx)
            return INVALID;
        if (j < 0 || j >= ny)
            return INVALID;
        if (k < 0 || k >= nz)
            return INVALID;

        return i + j * nx + k * nx * ny;
    }

    GPU_FUNC inline int getIndex(Coord pos)
    {
        int i = floorf((pos[0] - lo[0]) / ds);
        int j = floorf((pos[1] - lo[1]) / ds);
        int k = floorf((pos[2] - lo[2]) / ds);

        return getIndex(i, j, k);
    }

    GPU_FUNC inline int3 getIndex3(Coord pos)
    {
        int i = floorf((pos[0] - lo[0]) / ds);
        int j = floorf((pos[1] - lo[1]) / ds);
        int k = floorf((pos[2] - lo[2]) / ds);

        return make_int3(i, j, k);
    }

    GPU_FUNC inline int getCounter(int gId)
    {
        if (gId >= num - 1)
        {
            return particle_num - index[gId];
        }
        return index[gId + 1] - index[gId];
        //return counter[gId];
    }

    GPU_FUNC inline int getParticleId(int gId, int n)
    {
        return ids[index[gId] + n];
    }

public:
    int num;
    int nx, ny, nz;

    int particle_num = 0;

    Real ds;

    Coord lo;
    Coord hi;

    //int npMax;        //maximum particle number for each cell

    int* ids     = nullptr;
    int* counter = nullptr;
    int* index   = nullptr;

    Scan*           m_scan   = nullptr;
    Reduction<int>* m_reduce = nullptr;
};

#ifdef PRECISION_FLOAT
template class GridHash<DataType3f>;
#else
template class GridHash<DataType3d>;
#endif
}  // namespace PhysIKA
