#pragma once

#ifndef SANDINTERACTIONFORCESOLVER_H
#define SANDINTERACTIONFORCESOLVER_H

#include "Core/Array/Array.h"
#include "Core/Array/DynamicArray.h"
#include "Dynamics/RigidBody/PBDRigid/PointSDFContactDetector.h"

#include "Core/Utility/Reduction.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Dynamics/ParticleSystem/Kernel.h"

#include <unordered_map>

namespace PhysIKA {
class SandInteractionForceSolver
{
public:
    //void setSDFs(std::shared_ptr< std::vector<DistanceField3D<DataType3f>>> sdfs) { m_sdfs = sdfs; }
    //std::shared_ptr< std::vector<DistanceField3D<DataType3f>>> getSDFs()const { return m_sdfs; }
    const std::unordered_map<int, DistanceField3D<DataType3f>>& getSDFs() const
    {
        return m_sdfMap;
    }

    void addSDF(DistanceField3D<DataType3f>& sdf, int rigidid = -1);

    void updateSinkInfo(int i);

    //void computeBuoyancy();
    void computeSingleBuoyance(int i, Real dt);

    void computeSingleDragForce(int i, Real dt);

    void computeParticleInteractVelocity(int i, Real dt);

    //void computeSingleDragForceFri(int i);

    //void computeSingleInteractionForce(int i, Real dt, Vector3d& force, Vector3d& torque);

    void compute(Real dt);
    void computeSingleBody(int i, Real dt);

    bool collisionValid(RigidBody2_ptr prigid);

    void setPreBodyInfo();
    void updateBodyAverageVel(Real dt);

private:
    void _smoothVelocityChange();

    double _enlargerBuoyancy(double f, const Vector3d& t, double mass);

    double _minEng(const Vector3d& dF, const Vector3d& dT, double relvdf, int i, double dt);

    void _applyForceTorque(const Vector3d& F, const Vector3d& T, int i, Real dt);

    void _stableDamping(int i, Vector3d& F, Vector3d& T, Real dt);

    void _copyHostBodyToGPU(int i);

public:
    std::vector<RigidBody2_ptr>* m_prigids = 0;

    DeviceArray<PBDBodyInfo<double>>* m_body = 0;
    //std::vector<RigidBody2_ptr>* m_hostRigid = 0;
    PBDBodyInfo<double>* m_hostBody = 0;

    DeviceDArray<Vector3d>* m_particlePos  = 0;
    DeviceDArray<double>*   m_particleMass = 0;
    DeviceDArray<Vector3d>* m_particleVel  = 0;

    DeviceHeightField1d* m_land = 0;

    //NeighborList<int>* m_neighbor = 0;
    NeighborField<int> m_neighbor;

    double m_smoothLength   = 0.1;
    double m_sampleSize     = 0.005;
    double m_buoyancyFactor = 1.0;

    double m_gravity = 9.8;

    double m_beAddForceUpdate = true;

    double m_rho        = 1000.0;
    double m_e          = 0.0;
    double m_CsHorizon  = 0.15;  // Sand velocity coupling parameter, horizontal direction.
    double m_CsVertical = 0.5;   // Sand velocity coupling parameter, vertical direction.
    double m_Cprob      = 10000;

    double m_alpha = 15;
    double m_beta  = 0.6;
    double m_Cdrag = 1.0;  // Rigid drag force interaction paramter.

    double m_sandMu = 0.8;

    bool m_useStickParticleVelUpdate = false;

    int m_sandCollisionGroup = 1;
    int m_sandCollisionMask  = 1;

private:
    //std::shared_ptr<std::vector<DistanceField3D<DataType3f>>> m_sdfs;
    std::unordered_map<int, DistanceField3D<DataType3f>> m_sdfMap;
    //std::shared_ptr<double>

    DeviceDArray<Vector3d> m_buoyancyF;
    DeviceDArray<Vector3d> m_buoyancyT;
    double                 m_Abuo = 1.0;

    DeviceDArray<double>   m_devArr1d;
    DeviceDArray<Vector3d> m_devArr3d;

    DeviceDArray<Vector3d> m_dragF;
    DeviceDArray<Vector3d> m_dragT;

    DeviceDArray<double>   m_relvDf;
    DeviceDArray<Vector3d> m_dForce;
    DeviceDArray<Vector3d> m_dTorque;

    DeviceDArray<Vector3d> m_dVel;

    HostDArray<double>   m_hostArr1d;
    HostDArray<Vector3d> m_hostArr3d;

    DeviceDArray<double>   m_topH;
    DeviceDArray<double>   m_botH;
    DeviceDArray<Vector3d> m_topNormal;
    DeviceDArray<Vector3d> m_botNormal;

    Reduction<double>     m_reductiond;
    SpikyKernel2D<double> m_kernel;

    HostDArray<PBDBodyInfo<double>>   m_prevBody;
    DeviceDArray<PBDBodyInfo<double>> m_averageBodyInfo;
};
}  // namespace PhysIKA

#endif  // PBDSANDRIGIDINTERACTION_H