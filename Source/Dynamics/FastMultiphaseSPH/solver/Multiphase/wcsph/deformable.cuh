#pragma once

#include "SimDataMultiphase.h"

namespace msph {
void computeInnerForce_host(int num_particles);
void updateParticleState_host(int num_particles);
void correctKernel_host(int num_particles);
};  // namespace msph