#pragma once
#ifndef _PBDSANDSOLVER_H
#define _PBDSANDSOLVER_H

#include "Dynamics/Sand/SandSolverInterface.h"

#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Dynamics/ParticleSystem/Kernel.h"

#include "Framework/Topology/NeighborList.h"
#include "Framework/Topology/NeighborQuery.h"

#include "Core/Array/DynamicArray.h"

#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/DeclareModuleField.h"

#include <functional>

namespace PhysIKA {

template class NeighborQuery<DataType3d>;

enum ParticleType
{
    SAND,
    SANDINACTIVE,
    BOUNDARY
};

class PBDSandSolver : public SandSolverInterface
{
public:
    typedef std::function<void(PBDSandSolver*)> PostInitFun;

public:
    PBDSandSolver();

    virtual bool initialize();

    virtual bool stepSimulation(float dt);

    virtual float getMaxTimeStep() override
    {
        return 0.02;
    }

    virtual bool forwardSubStep(float dt);
    virtual bool velocityUpdate(float dt);
    virtual bool positionUpdate(float dt);
    virtual bool infoUpdate(float dt);

    virtual void freeFlow(int steps = 100);

    virtual void updateUserParticle(DeviceArray<Vector3f>& usePoints) override;

    void setParticles(Vector3d* pos, Vector3d* vel, double* mass, ParticleType* particleType, int num, double rho0, double m0, double smoothLen, double mu, double h0);

    void setLand(HostHeightField1d& land);

    void setHeight(HostHeightField1d& height);

    void computeSandHeight();

    void computeSandStaticHeight();

    //void updateParticleMass(double dt);

    DeviceDArray<Vector3d>& getParticleVelocity()
    {
        return m_particleVel;
    }
    const DeviceDArray<Vector3d>& getParticleVelocity() const
    {
        return m_particleVel;
    }

    DeviceDArray<Vector3d>& getParticlePosition()
    {
        return m_particlePos;
    }
    const DeviceDArray<Vector3d>& getParticlePosition() const
    {
        return m_particlePos;
    }

    DeviceDArray<Vector3d>& getParticlePosition3D()
    {
        return m_particlePos3D;
    }
    const DeviceDArray<Vector3d>& getParticlePosition3D() const
    {
        return m_particlePos3D;
    }

    DeviceDArray<double>& getParticleRho2D()
    {
        return m_particleRho2D;
    }
    const DeviceDArray<double>& getParticleRho2D() const
    {
        return m_particleRho2D;
    }

    DeviceDArray<double>& getParticleMass()
    {
        return m_particleMass;
    }
    const DeviceDArray<double>& getParticleMass() const
    {
        return m_particleMass;
    }

    DeviceDArray<ParticleType>& getParticleTypes()
    {
        return m_particleType;
    }
    const DeviceDArray<ParticleType>& getParticleTypes() const
    {
        return m_particleType;
    }

    NeighborField<int>& getNeighborField()
    {
        return m_neighbors;
    }
    NeighborList<int>& getNeighborList()
    {
        return m_neighbors.getValue();
    }
    //const NeighborList<int>& getNeighborList() const { return m_neighbors.getValue(); }

    DeviceHeightField1d& getLand()
    {
        return m_land;
    }
    const DeviceHeightField1d& getLand() const
    {
        return m_land;
    }

    double getRho0()
    {
        return m_rho0;
    }
    double getSmoothingLength()
    {
        return m_smoothingLength.getValue();
    }

    double getM0()
    {
        return m_m0;
    }

    void needPosition3D(bool needpos3d)
    {
        m_need3DPos = needpos3d;
    }

    void setFlowingLayerHeight(double h)
    {
        m_flowingLayerHeight = h;
    }
    double getFlowingLayerHeight() const
    {
        return m_flowingLayerHeight;
    }

    void setPostInitializeFun(PostInitFun fun)
    {
        m_postInitFun = fun;
    }

private:
    void _updateRawParticleRho();

    void _updateParticlePos3D();

    void _updateGridHeight();

    void _initBoundaryParticle();

    void _doNeighborDetection();
    void _updateGridHash();

    void _updateStaticHeightChange(double dt);

    void _generateAndEliminateParticle(double dt);

    void _particleNumResize(int n);
    void _particleNumReserve(int n);

public:
    //DEF_EMPTY_VAR(CFL, double, "CFL Condition Number");
    double m_CFL = 0.5;

    double m_flowingLayerHeight = 0.1;

private:
    std::shared_ptr<PBDSolver> m_pbdsolver;

    DeviceHeightField1d m_land;

    DeviceHeightField1d m_height;
    DeviceHeightField1d m_staticHeight;

    DeviceHeightField1d m_dStaticHeight;

    //DeviceHeightField1d m_gridRho;
    //DeviceHeightField3d m_gridVel;

    DeviceDArray<Vector3d> m_prePosition;
    DeviceDArray<Vector3d> m_particlePos;
    DeviceDArray<Vector3d> m_particlePos3D;

    DeviceDArray<Vector3d> m_dPosition;
    DeviceDArray<Vector3d> m_particleVel;
    DeviceDArray<double>   m_particleMass;
    DeviceDArray<double>   m_particleRho2D;
    DeviceDArray<double>   m_lambda;

    DeviceDArray<ParticleType> m_particleType;

    SpikyKernel2D<double> m_kernel;

    //

    //NeighborList<int> m_neighbors;
    std::shared_ptr<NeighborQuery<DataType3f>> m_neighborQuery;

    NeighborField<int>         m_neighbors;
    DeviceArrayField<Vector3f> m_position;
    VarField<float>            m_smoothingLength;

    GridHash<DataType3f> m_gridParticleHash;

    DeviceArray<int> m_genEliCount;
    int              m_neighborSize = 2;
    //double m_smoothLength;
    double m_rho0 = 1000.0;
    double m_m0;
    int    m_subStepNum = 4;

    double m_mu = 0.1;  // 0.12;
    double m_h0 = 0.1;

    bool m_need3DPos = false;

    PostInitFun m_postInitFun;
};

}  // namespace PhysIKA

#endif  //_PBDSANDSOLVER_H