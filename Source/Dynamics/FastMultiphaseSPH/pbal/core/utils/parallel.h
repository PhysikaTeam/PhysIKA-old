#pragma once

#include <core/math/vec.h>
#include <tbb/parallel_for.h>

namespace pbal {

template <typename Op>
void forEachIndex(int size, Op op)
{
    for (int idx = 0; idx < size; idx++)
    {
        op(idx);
    }
}

template <typename Op>
void parallelForEachIndex(int size, Op op)
{
    tbb::parallel_for(
        0, size, [&](int idx) {
            op(idx);
        });
}

template <typename Op>
void forEachIndex(Size2 size, Op op)
{
    for (int j = 0; j < size.y; j++)
    {
        for (int i = 0; i < size.x; i++)
        {
            op(i, j);
        }
    }
}

template <typename Op>
void forEachIndex(Size3 size, Op op)
{
    for (int k = 0; k < size.z; k++)
    {
        for (int j = 0; j < size.y; j++)
        {
            for (int i = 0; i < size.x; i++)
            {
                op(i, j, k);
            }
        }
    }
}

template <typename Op>
void parallelForEachIndex(Size2 size, Op op)
{
    int n = size.x * size.y;
    tbb::parallel_for(
        0, n, [&](int idx) {
            int i = idx / size.y;
            int j = idx % size.y;

            op(i, j);
        });
}

template <typename Op>
void parallelForEachIndex(Size3 size, Op op)
{
    int n = size.x * size.y * size.z;
    tbb::parallel_for(
        0, n, [&](int idx) {
            int i = idx / (size.y * size.z);
            int j = (idx / size.z) % size.y;
            int k = idx % size.z;

            op(i, j, k);
        });
}

}  // namespace pbal