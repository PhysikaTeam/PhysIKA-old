#pragma once

#include <core/math/vec.h>

#include <core/scene/colliders/collider2.h>
#include <core/scene/emitters/emitter2.h>

#include <memory>
#include <string>
#include <vector>

namespace pbal {

struct Scene2
{
public:
    Vec2d spacing;
    // bounding box
    Vec2d lowerCorner;
    Vec2d upperCorner;
    // sdf
    GridScalar2d emitterSdf;

    std::vector<Collider2Ptr> colliders;
    std::vector<Emitter2Ptr>  emitters;

    void update(double dt)
    {
        for (auto& collider : colliders)
        {
            collider->update(dt);
        }
        for (auto& emitter : emitters)
        {
            emitter->update(dt);
        }
    }
};

typedef std::shared_ptr<Scene2> Scene2Ptr;

}  // namespace pbal