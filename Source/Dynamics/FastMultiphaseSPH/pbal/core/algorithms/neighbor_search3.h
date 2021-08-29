#pragma once

#include <core/math/vec.h>

#include <algorithm>
#include <vector>
#include <core/utils/parallel.h>

namespace pbal {
template <typename real>
struct UnboundedGrid3
{
    using Cell   = Size3;
    using HashId = unsigned int;
    Vec<real, 3> spacingInv;
    UnboundedGrid3(Vec<real, 3> spacing)
        : spacingInv(1 / spacing.x, 1 / spacing.y, 1 / spacing.z) {}
    Cell cellFromPos(Vec<real, 3> p) const
    {
        Vec<real, 3> offset = p * spacingInv;
        Cell         cell;
        cell.x = floor(offset.x);
        cell.y = floor(offset.y);
        cell.z = floor(offset.z);
        return cell;
    }
    HashId hashFromCell(Cell c) const
    {
        return interleave3(c.x) | interleave3(c.y) << 1 | interleave3(c.z) << 2;
    }
    static unsigned int interleave3(unsigned short u)
    {
        unsigned int x = u;
        x              = (x | (x << 16)) & 0xff0000ff;
        x              = (x | (x << 8)) & 0x0f00f00f;
        x              = (x | (x << 4)) & 0xc30c30c3;
        x              = (x | (x << 2)) & 0x49249249;
        return x;
    }
};

template <typename real>
class NeighborSearch3 : public UnboundedGrid3<real>
{
public:
    using Cell   = typename UnboundedGrid3<real>::Cell;
    using HashId = typename UnboundedGrid3<real>::HashId;
    using UnboundedGrid3<real>::cellFromPos;
    using UnboundedGrid3<real>::hashFromCell;

private:
    std::vector<HashId> hashes;
    std::vector<size_t> offsets;
    std::vector<size_t> ids;

public:
    NeighborSearch3(const Vec<real, 3>& spacing)
        : UnboundedGrid3<real>(spacing) {}

    void sort(size_t ptsNum, const Vec<real, 3>* pos)
    {
        using ParticlePack = struct
        {
            HashId hashId;
            size_t particleId;
        };
        hashes.clear();
        offsets.clear();
        std::vector<ParticlePack> sortBuffer(ptsNum);
        parallelForEachIndex(ptsNum,
                             [&](int idx) {
                                 auto p                     = pos[idx];
                                 sortBuffer[idx].hashId     = hashFromCell(cellFromPos(p));
                                 sortBuffer[idx].particleId = idx;
                             });
        // sort
        std::sort(sortBuffer.begin(), sortBuffer.end(), [](const ParticlePack& a, const ParticlePack& b) {
            return a.hashId < b.hashId;
        });
        // counting ocupied hash cells
        ids.resize(ptsNum);
        for (size_t i = 0; i < ptsNum; i++)
        {
            ids[i] = sortBuffer[i].particleId;
        }
        if (ptsNum > 0)
        {
            HashId prev = sortBuffer[0].hashId;
            hashes.push_back(prev);
            offsets.push_back(0);
            for (size_t i = 1; i < ptsNum; i++)
            {
                HashId c = sortBuffer[i].hashId;
                if (prev != c)
                {
                    prev = c;
                    hashes.push_back(prev);
                    offsets.push_back(i);
                }
            }
            offsets.push_back(ptsNum);
        }
    }
    void sort(const std::vector<Vec<real, 3>> pos)
    {
        sort(pos.size(), pos.data());
    }
    size_t* getPermutation()
    {
        return ids.data();
    }
    template <typename Op>
    void forEachIndexInCell(
        Cell cell,
        Op   op) const
    {
        unsigned int hashID     = hashFromCell(cell);
        size_t       hashOffset = std::lower_bound(hashes.begin(), hashes.end(), hashID) - hashes.begin();
        if (hashOffset < hashes.size() && hashes[hashOffset] == hashID)
        {
            const size_t begin = offsets[hashOffset];
            const size_t end   = offsets[hashOffset + 1];
            for (size_t i = begin; i < end; i++)
                op(ids[i]);
        }
    }
    template <typename Op>
    void forEachNeighborIndex(
        Vec3d p,
        Op    op) const
    {
        const Cell cell = UnboundedGrid3<real>::cellFromPos(p);
        Cell       min  = cell - 1;
        Cell       max  = cell + 1;
        for (int z = min.z; z <= max.z; z++)
            for (int y = min.y; y <= max.y; y++)
                for (int x = min.x; x <= max.x; x++)
                {
                    Cell cell = Cell(x, y, z);
                    forEachIndexInCell(cell, op);
                }
    }
};
using NeighborSearch3f = NeighborSearch3<float>;
using NeighborSearch3d = NeighborSearch3<double>;
}  // namespace pbal
