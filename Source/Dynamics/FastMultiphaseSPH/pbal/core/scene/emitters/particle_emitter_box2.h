#pragma once

#include "emitter2.h"

#include <string>
#include <vector>

#include <core/math/vec.h>
#include <core/particles/particle_system2.h>

namespace pbal {

struct ParticleEmitterBox2 : Emitter2
{
public:
    Vec2d  center;
    Vec2d  distance;
    double r;

    bool emitOnce = true;

    const ParticleSystem2Ptr& getTarget()
    {
        return _target;
    }

    void setTarget(const ParticleSystem2Ptr& target)
    {
        _target = target;
    }

    void update(double dt) override
    {
        static bool hasEmit = false;
        if (emitOnce && hasEmit)
        {
            return;
        }
    }

private:
    ParticleSystem2Ptr _target;
};

}  // namespace pbal