#pragma once

#ifndef _SANDVISUALPOINTSAMPLEMODULE_H
#define _SANDVISUALPOINTSAMPLEMODULE_H

#include "Core/Vector/vector_3d.h"
#include "Core/Array/Array3D.h"

#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/ModuleCustom.h"

#include "Framework/Framework/DeclareModuleField.h"
#include "Dynamics/Sand/PBDSandSolver.h"

namespace PhysIKA {

/**
    * @brief Paritcle sampler for height field.
    */
class SandHeightRenderParticleSampler : public CustomModule
{
public:
    SandHeightRenderParticleSampler();
    ~SandHeightRenderParticleSampler();

    /**
        * @brief Initialize sampler data.
        * @note Please set correct land height pointer and sand height point before calling this function.
        */
    void Initalize(int nx, int ny, int freq, int layer, float gridLength);

    void Generate();

    //void update() { compute(); }

    //void doSampling(Vector3f* pointSample, DeviceArrayPitch2D4f& sandGrid);

    //void doSampling(Vector3f* pointSample, DeviceHeightField1d& sandHeight, DeviceHeightField1d& landHeight);

    int sampleCount()
    {
        return m_nx * m_ny * var_SampleFreq.getValue() * var_SampleFreq.getValue() * var_SampleLayer.getValue();
    }

    //void Advect(float4* g_vel, int g_nx, int g_ny, int pitch, float g_spacing, float dt);

    //void Deposit();

    virtual void compute();

    virtual void applyCustomBehavior()
    {
        compute();
    }

    virtual bool initializeImpl() override
    {
        compute();
        return true;
    }

public:
    DEF_EMPTY_VAR(SampleLayer, int, "Verticle sample layer number.");
    DEF_EMPTY_VAR(SampleFreq, int, "Horizontal sample frequency.");

    //DEF_EMPTY_VAR(Spacing, float, "Sample space.");

    DeviceHeightField1d* m_sandHeight = 0;
    DeviceHeightField1d* m_landHeight = 0;

public:
    int m_nx;
    int m_ny;

    //int m_sampleLayer;
    //int m_sampleFreq;
    float m_spacing;

    float m_gridLength;

    DeviceArray3D<float3> m_normalizePosition;
};

/**
    * @brief Particle sampler for particle based sand.
    */
class ParticleSandRenderSampler : public CustomModule
{
public:
    ParticleSandRenderSampler() {}
    ~ParticleSandRenderSampler() {}

    void Initialize(std::shared_ptr<PBDSandSolver> solver);

    virtual void compute();
    //void update() { compute(); }

    virtual void applyCustomBehavior()
    {
        compute();
    }

    virtual bool initializeImpl() override
    {
        compute();
        return true;
    }

public:
    DeviceDArray<ParticleType>* particleType = 0;
    DeviceHeightField1d*        landHeight   = 0;
    DeviceDArray<Vector3d>*     particlePos  = 0;

    DeviceDArray<double>* particleRho2D = 0;
    int                   rho0          = 1000.0;

private:
};

}  // namespace PhysIKA

#endif  // _SANDVISUALPOINTSAMPLEMODULE_H