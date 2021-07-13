#pragma once

#include <memory>
#include <string>
#include <vector>

#include <core/math/vec.h>

namespace pbal {

struct Collider2
{
public:
    std::string  type;
    virtual void update(double dt) = 0;
};

typedef std::shared_ptr<Collider2> Collider2Ptr;

}  // namespace pbal