#pragma once

#ifndef PHYSIKA_PBDSOLVER_H
#define PHYSIKA_PBDSOLVER_H

#include "Core/Array/Array.h"
#include "Core/Array/DynamicArray.h"

#include "Dynamics/RigidBody/PBDRigid/PBDJoint.h"
#include "Core/Utility/CTimer.h"
#include "Dynamics/RigidBody/ContactInfo.h"

#include "Framework/Topology/NeighborList.h"

#include "Dynamics/ParticleSystem/Kernel.h"

#include <functional>

#include <vector>

namespace PhysIKA {

class PBDSolver
{
    //DECLARE_CLASS_1(PBDSolverNode, TReal)
public:
    typedef std::function<void(PBDSolver*, Real)> DetectionFun;

    typedef std::function<void(Real)> CustomUpdateFun;

public:
    PBDSolver();
    void setUseGPU(bool usegpu = true)
    {
        m_useGPU = usegpu;
    }

    virtual bool initialize();

    virtual void advance(Real dt);

    // ********  Deplicated  ********
    virtual void CPUUpdate(Real dt);
    virtual void GPUUpdate(Real dt);
    virtual void integrateBodyForce(Real dt);
    virtual void integrateBodyVelocity(Real dt);
    virtual void solveSubStepGPU(Real dt);

    /**
        * @brief Integrate velocity and position
        */
    //virtual void integrateMotion(Real dt);

    /**
        * @brief Solve position constraints.
        */
    //virtual void solvePosition(Real dt);

    /**
        * @brief Step forward one time step. GPU version.
        * @details Contact constraints are solved using only 1 iteration.
        */
    virtual void forwardSubStepGPU(Real dt);
    /**
        * @brief Step forward one time step. GPU version.
        * @details Contact constraints are solved using m_numContactSolveIter iteration.
        */
    virtual void forwardSubStepGPU2(Real dt);

    /**
        * @brief Step forward one time step. CPU version.
        * @details Contact constraints are solved using only 1 iteration.
        */
    virtual void forwardSubStepCPU(Real dt);
    /**
        * @brief Step forward one time step. CPU version.
        * @details Contact constraints are solved using m_numContactSolveIter iteration.
        */
    virtual void forwardSubStepCPU2(Real dt);

    /**
        * @brief Step forward one time step. 
        */
    virtual void forwardSubStep(Real dt);

    /**
        * @brief Call user define update functions.
        */
    virtual void doCustomUpdate(Real dt);

    int addRigid(RigidBody2_ptr prigid);

    int  addPBDRigid(const PBDBodyInfo<double>& pbdbody);
    void setBodyDirty()
    {
        m_bodyDataDirty = true;
    }

    /**
        * @brief Add permanent joints.
        * @details Note, all permanent joints should be added before the first frame.
        */
    int  addPBDJoint(const PBDJoint<double>& pbdjoint, int bodyid0, int bodyid1);
    bool setJointDirty()
    {
        m_jontDataDirty = true;
    }

    Array<PBDJoint<double>>& getGPUJoints()
    {
        return m_GPUJoints;
    }
    const Array<PBDJoint<double>>& getGPUJoints() const
    {
        return m_GPUJoints;
    }

    std::vector<RigidBody2_ptr>& getRigidBodys()
    {
        return m_rigids;
    }
    const std::vector<RigidBody2_ptr>& getRigidBodys() const
    {
        return m_rigids;
    }

    std::vector<PBDBodyInfo<double>>& getCPUBody()
    {
        return m_CPUBodys;
    }
    const std::vector<PBDBodyInfo<double>>& getCPUBody() const
    {
        return m_CPUBodys;
    }

    Array<PBDBodyInfo<double>>& getGPUBody()
    {
        return m_GPUBodys;
    }
    const Array<PBDBodyInfo<double>>& getGPUBody() const
    {
        return m_GPUBodys;
    }

    /**
        * @brief Update rigid body info 
        */
    void updateRigidToPBDBody();
    void updateRigidToCPUBody();
    void updateRigidToGPUBody();
    //void updateRigidToCPU
    void updateCPUToGPUBody();
    void updateGPUToCPUBody();

    /**
        * @brief Integrate body force and velocity.
        * @note Deplicated function.
        */
    void integrateBody(Real dt);

    /**
        * @brief Solve joint constraints.
        * @note Deplicated function.
        */
    void solveJoints(Real dt);

    /**
        * @brief Set body constact pair joints.
        * @details Input contacts are GPU data.
        */
    void setContactJoints(DeviceDArray<ContactInfo<double>>& contacts, int nContact);

    void setNarrowDetectionFun(DetectionFun fun)
    {
        m_narrowPhaseDetection = fun;
    }

    void setBroadDetectionFun(DetectionFun fun)
    {
        m_broadPhaseDetection = fun;
    }

    void synFromBodiedToRigid();

    void addCustomUpdateFunciton(CustomUpdateFun fun)
    {
        m_selfupdate.push_back(fun);
    }

    void enableVelocitySolve()
    {
        m_solveVelocity = true;
    }
    void disableVelocitySolve()
    {
        m_solveVelocity = false;
    }

private:
    /**
        * @brief Synchronize body data from RigidBody2 object to PBDBodyInfo.
        */
    void _toPBDBody(RigidBody2_ptr prigid, PBDBodyInfo<double>& pbdbody);

    /**
        * @brief Synchronize body data from PBDBodyInfo to RigidBody2.
        */
    void _fromPBDBody(const PBDBodyInfo<double>& pbdbody, RigidBody2_ptr prigid);

    void _updateBodyPointerInJoint();

    void _initGPUData();

    void _onRigidDataDirty();

public:
    Array<PBDBodyInfo<double>> m_GPUBodys;
    Array<PBDJoint<double>>    m_GPUJoints;
    Array<Vector3d>            m_GPUPosChange;
    Array<Quaterniond>         m_GPURotChange;
    Array<Vector3d>            m_GPULinvChange;
    Array<Vector3d>            m_GPUAngvChange;
    Array<int>                 m_GPUConstraintCount;
    Array<double>              m_GPUOmega;

    //Array<PBDBodyInfo<double>, DeviceType::CPU> m_CPUBodys;
    //Array<PBDJoint<double>, DeviceType::CPU> m_CPUJoints;
    std::vector<PBDBodyInfo<double>> m_CPUBodys;
    std::vector<PBDJoint<double>>    m_CPUJoints;

    int m_nBodies          = 0;
    int m_nPermanentBodies = 0;
    int m_nJoints          = 0;
    int m_nPermanentJoints = 0;

    std::vector<RigidBody2_ptr> m_rigids;

    //Real m_minSubTimestep;
    int  m_numContactSolveIter = 1;
    int  m_numSubstep;
    bool m_useGPU;
    bool m_solveVelocity = true;

    int m_blockdim = 512;
    //int m_numSubStep = 20;

    CTimer m_timer;
    double m_totalTime     = 0.0;
    int    m_totalFrame    = 0;
    double m_detectionTime = 0.0;

    bool m_jontDataDirty = true;
    bool m_bodyDataDirty = true;

    double collisionThreashold = 0.01;

    DetectionFun m_narrowPhaseDetection;
    DetectionFun m_broadPhaseDetection;

    // Vector of user defined update function.
    // Those functions will be call after position constraints are solved.
    std::vector<CustomUpdateFun> m_selfupdate;
};

/**
    * @brief Solver for collision and contact constraint between particle and normal (rigid) body.
    * @details All joints are contact joints in this solver, and body1 of joints should be particles.
    */
class PBDParticleBodyContactSolver
{
public:
    PBDParticleBodyContactSolver();

    ~PBDParticleBodyContactSolver() {}

    virtual void forwardSubStep(Real dt, bool updateVel = true);

    void solveCollisionVelocity(Real dt);

    void updateParticleBodyInfo(double mu);

    bool buildJoints();

    //bool buildParticleBody(DeviceDArray<Vector3d>& parPos/*, DeviceDArray<Vector3d>& */);

public:
    DeviceArray<PBDBodyInfo<double>>* m_body = 0;

    DeviceDArray<Vector3d>*           m_particlePos  = 0;
    DeviceDArray<double>*             m_particleMass = 0;
    DeviceArray<PBDBodyInfo<double>>* m_particle     = 0;

    DeviceDArray<ContactInfo<double>>* m_contacts = 0;

    // Particle rigid body contact joints.
    // body0 of joint is rigid body.
    // body1 of joint is particle.
    DeviceDArray<PBDJoint<double>>* m_joints = 0;

    DeviceDArray<Vector3d>* m_particleVel = 0;

    DeviceDArray<Vector3d>    m_bodyPosChange;
    DeviceDArray<Quaterniond> m_bodyRotChange;

    DeviceDArray<Vector3d> m_bodyLinvChange;
    DeviceDArray<Vector3d> m_bodyAngvChange;

    double m_omega = 1.0;
    //DeviceDArray<
    // HostArray<PBDArray
};

class PBDDensitySolver2D
{
public:
    void forwardOneSubStep(Real dt);

public:
public:
    //DeviceArray<PBDBodyInfo<double>>* m_particle = 0;
    DeviceDArray<Vector3d>* m_particlePos = 0;

    DeviceDArray<Vector3d>* m_particleVel = 0;

    DeviceDArray<double>* m_particleRho2D = 0;

    DeviceDArray<double>* m_particleMass = 0;

    //NeighborList<int>* m_neighbor = 0;
    NeighborField<int> m_neighbor;

    SpikyKernel2D<double> m_kernel;
    double                m_smoothLength = 0.0;
    double                m_rho0         = 1000.0;
    double                minh           = 0.01;

    DeviceDArray<double> m_lambda;

    DeviceDArray<double> m_targetRho;

    DeviceDArray<Vector3d> m_particleDpos;
};

}  // namespace PhysIKA

#endif  // PHYSIKA_PBDSOLVER_H