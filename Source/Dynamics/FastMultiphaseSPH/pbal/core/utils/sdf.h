#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

#include <core/math/vec.h>

namespace pbal {

template <typename T>
T boxSdf(const Vec2<T>& position, const Vec2<T>& centre, const Vec2<T>& b)
{
    auto p = position - centre;
    auto d = Vec2<T>(std::abs(p.x), std::abs(p.y)) - b;
    return std::min(std::max(d.x, d.y), T())
           + Vec2<T>(std::max(d.x, T()), std::max(d.y, T())).length();
};

template <typename T>
T boxSdf(const Vec3<T>& position, const Vec3<T>& centre, const Vec3<T>& b)
{
    auto p = position - centre;
    auto d = Vec3<T>(std::abs(p.x), std::abs(p.y), std::abs(p.z)) - b;
    return std::min(std::max(std::max(d.x, d.y), d.z), T())
           + Vec3<T>(std::max(d.x, T()), std::max(d.y, T()), std::max(d.z, T())).length();
};

}  // namespace pbal