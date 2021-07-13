#pragma once

#include <vector>
#include "Dynamics/RigidBody/BoxAABB3d.h"

namespace PhysIKA {
template <typename T>
class SortSweepDetector
{
public:
    SortSweepDetector() {}

    void reset();

    bool detect(const std::vector<BoxAABB3d<T>>& boxes, std::vector<std::pair<int, int>>& collision_pairs);

    void updateAxis(const std::vector<BoxAABB3d<T>>& boxes);

private:
    int m_n = 0;

    int m_sort_axis = 0;

    // pair: first -- interval id, it will be negative if the value is the beginning of the interval;
    // pair: second -- beginning or end value of the interval
    std::vector<std::pair<int, T>> m_sorted_pair;
    //std::vector<bool> m_is_begin_x;

    //std::vector<std::pair<int, T>> m_sorted_y;
    ////std::vector<bool> m_is_begin_y;
    //
    //std::vector<std::pair<int, T>> m_sorted_z;
    ////std::vector<bool> m_is_begin_z;
};

#ifdef PRECISION_FLOAT
template class SortSweepDetector<float>;
#else
template class SortSweepDetector<double>;
#endif
}  // namespace PhysIKA