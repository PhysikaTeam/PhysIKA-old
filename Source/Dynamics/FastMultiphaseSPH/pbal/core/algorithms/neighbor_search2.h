#pragma once

#include <core/math/vec.h>

#include <algorithm>
#include <vector>
#include <core/utils/parallel.h>

namespace pbal {
template <typename real>
struct UnboundedGrid2
{
    using Cell   = Size2;
    using HashId = unsigned int;
    Vec<real, 2> spacingInv;
    UnboundedGrid2() {}
    UnboundedGrid2(Vec<real, 2> spacing)
        : spacingInv(1 / spacing.x, 1 / spacing.y) {}
    Cell cellFromPos(Vec<real, 2> p) const
    {
        Vec<real, 2> offset = p * spacingInv;
        Cell         cell;
        cell.x = floor(offset.x);
        cell.y = floor(offset.y);
        return cell;
    }
    HashId hashFromCell(Cell c) const
    {
        return interleave2(c.x) | interleave2(c.y) << 1;
    }
    static unsigned int interleave2(unsigned short u)
    {
        unsigned int x = u;
        x              = (x | (x << 8)) & 0x00ff00ff;
        x              = (x | (x << 4)) & 0x0f0f0f0f;
        x              = (x | (x << 2)) & 0x33333333;
        x              = (x | (x << 1)) & 0x55555555;
        return x;
    }
};

template <typename real>
class NeighborSearch2 : public UnboundedGrid2<real>
{
public:
    using Cell   = typename UnboundedGrid2<real>::Cell;
    using HashId = typename UnboundedGrid2<real>::HashId;
    using UnboundedGrid2<real>::cellFromPos;
    using UnboundedGrid2<real>::hashFromCell;
    std::vector<HashId> hashes;
    std::vector<size_t> offsets;
    std::vector<size_t> ids;

public:
    NeighborSearch2() {}

    NeighborSearch2(const Vec<real, 2>& spacing)
        : UnboundedGrid2<real>(spacing) {}

    void sort(size_t ptsNum, const Vec<real, 2>* pos)
    {
        using ParticlePack = struct
        {
            HashId hashId;
            size_t particleId;
        };
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
    void sort(const std::vector<Vec<real, 2>> pos)
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
        Vec<real, 2> p,
        Op           op) const
    {
        const Cell cell = cellFromPos(p);
        Cell       min  = cell - 1;
        Cell       max  = cell + 1;
        for (int y = min.y; y <= max.y; y++)
            for (int x = min.x; x <= max.x; x++)
            {
                Cell cell = Cell(x, y);
                forEachIndexInCell(cell, op);
            }
    }
};
using NeighborSearch2f = NeighborSearch2<float>;
using NeighborSearch2d = NeighborSearch2<double>;
}  // namespace pbal
