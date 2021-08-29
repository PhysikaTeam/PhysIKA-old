#pragma once

#include <core/animation/frame.h>

namespace pbal {

class Animation
{
public:
    Animation() {}

    virtual ~Animation() {}

    void update(const Frame& frame)
    {
        onUpdate(frame);
    }

protected:
    virtual void onUpdate(const Frame& frame) = 0;
};

}  // namespace pbal
