#pragma once

#include "Dynamics/Sand/ParticleSandRigidInteraction.h"
#include <curand.h>

#include "Framework/Framework/ModuleCustom.h"

#ifndef DEMOCALLBACKS_H
#define DEMOCALLBACKS_H

namespace PhysIKA {
class ParticleGenerationCallback
{
public:
    void init(float xmin, float xmax, float zmin, float zmax, float m0, float rate, int maxNum = 100, long seed = 2341747);
    void handle(ParticleSandRigidInteraction* interactNode, float dt);

public:
    float gxMin = 0.f, gxMax = 1.f;
    float gzMin = 0.f, gzMax = 1.f;

    float generationRate = 1.0f;

    float particelMass = 1.0;

    int maxGenerationNum = 100;

private:
    curandState* devStates = 0;
};

class ParticleHeightOnZ  //:public CustomModule
{
public:
    ParticleHeightOnZ() {}
    ParticleHeightOnZ(DeviceDArray<ParticleType>* parType,
                      DeviceDArray<Vector3d>*     parPos,
                      DeviceDArray<double>*       parRho2D,
                      std::string                 filename,
                      double                      zval,
                      double                      w,
                      double                      rho)
    {
        particleType  = parType;
        particlePos   = parPos;
        particleRho2D = parRho2D;

        outputfilename = filename;
        zValue         = zval;
        searchWidth    = w;
        rho0           = rho;
    }

    virtual void handle(float dt);

public:
    double zValue = 0.0;

    double searchWidth = 0.05;
    double rho0        = 1000;

    DeviceDArray<ParticleType>* particleType  = 0;
    DeviceHeightField1d*        landHeight    = 0;
    DeviceDArray<Vector3d>*     particlePos   = 0;
    DeviceDArray<double>*       particleRho2D = 0;

    HostDArray<ParticleType> hostParType;
    HostDArray<Vector3d>     hostParPos;
    HostDArray<double>       hostParRho2D;

    std::string outputfilename;
};

}  // namespace PhysIKA
#endif  //DEMOCALLBACKS_H