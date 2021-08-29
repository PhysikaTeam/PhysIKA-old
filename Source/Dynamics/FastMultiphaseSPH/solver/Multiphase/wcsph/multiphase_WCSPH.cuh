#pragma once

#include "SimDataMultiphase.h"
#include "../../sort.h"

namespace msph {

void computeForce_host(int num_particles);
void advectParticles_host(int num_particles);

void UpdateParticleStateHost(int numParticles);

void computeDriftVel_host(int num_particles);
void computePhaseDiffusion_host(SimDataMultiphase& data, int num_particles, int num_types);
void computeInterPhaseTensor_host(int num_particles);

void updateMass_host(int num_particles);
void CopyParamToDevice(MultiphaseParam& h_param);
void CopyParamFromDevice(MultiphaseParam& h_param);
void copyDataPtrToDevice(const SimDataMultiphase& data);
void computeRigidVolume_host(int num_particles);

void BuildNeighborListHost(int num_particles);
};  // namespace msph