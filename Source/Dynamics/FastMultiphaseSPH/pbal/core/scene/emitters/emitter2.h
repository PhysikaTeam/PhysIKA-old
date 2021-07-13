#pragma once

#include <memory>
#include <vector>

#include <core/math/vec.h>

namespace pbal {

class Emitter2
{
public:
    std::string  type;
    virtual void update(double dt) = 0;
};

typedef std::shared_ptr<Emitter2> Emitter2Ptr;

}  // namespace pbal