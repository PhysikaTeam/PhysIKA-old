#pragma once

#include "emitter2.h"

#include <string>
#include <vector>

#include <core/math/vec.h>

namespace pbal {

struct EmitterParticles2 : Emitter2
{
    bool emitOnce = true;

    void update(double dt) override
    {
        static bool hasEmit = false;
        if (emitOnce && hasEmit)
        {
            return;
        }
    }
};

}  // namespace pbal