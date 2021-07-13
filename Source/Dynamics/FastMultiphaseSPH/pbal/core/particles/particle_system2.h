#pragma once

#include <core/math/vec.h>
#include <core/particles/particle2.h>

#include <memory>
#include <vector>

namespace pbal {

struct ParticleSystem2
{
    std::vector<Particle2> particles;
};

typedef std::shared_ptr<ParticleSystem2> ParticleSystem2Ptr;

}  // namespace pbal