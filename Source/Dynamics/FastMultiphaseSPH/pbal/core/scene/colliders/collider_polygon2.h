#pragma once

#include "collider2.h"

#include <string>
#include <vector>

#include <core/math/vec.h>

namespace pbal {

struct ColliderPolygon2 : Collider2
{
    typedef std::vector<Vec2d> Polygon;

    Polygon polygon;

    Vec2d  vel    = Vec2d();
    Vec2d  center = Vec2d();
    double w      = 0.0;

    void update(double dt) override
    {
        for (auto& pt : polygon)
        {
            auto n = pt - center;
            auto t = Vec2d(-n.y, n.x);
            auto v = vel + n * t;
            pt += v * dt;
        }
    }
};

}  // namespace pbal