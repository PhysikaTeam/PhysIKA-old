#pragma once

#ifndef PARTICLESANDRIGIDINTERACTION_H
#define PARTICLESANDRIGIDINTERACTION_H

#include "Dynamics/Sand/PBDSandSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PointSDFContactDetector.h"
#include "Framework/Framework/Node.h"
#include "Dynamics/Sand/SandInteractionForceSolver.h"

//#include "Dynamics/RigidBody/PBDRigid/B.h"
#include "Dynamics/RigidBody/PBDRigid/BodyContactDetector.h"

#include <functional>

namespace PhysIKA {
class ParticleSandRigidInteraction : public Node
{
public:
    typedef std::function<void(ParticleSandRigidInteraction*, Real)> CallbackFun;

public:
    ParticleSandRigidInteraction();
    ~ParticleSandRigidInteraction() {}

    virtual bool initialize() override;

    virtual void advance(Real dt) override;
    //void advect(Real dt);

    //void addSandChild()

    void setLandHeight(DeviceHeightField1d* landHeight)
    {
        m_landHeight = landHeight;
    }
    DeviceHeightField1d& getLandHeight()
    {
        return *m_landHeight;
    }

    void advectSubStep(Real dt);

    std::shared_ptr<PBDSandSolver> getSandSolver() const
    {
        return m_sandSolver;
    }
    void setSandSolver(std::shared_ptr<PBDSandSolver> sandSolver)
    {
        m_sandSolver = sandSolver;
    }

    std::shared_ptr<PBDSolver> getRigidSolver() const
    {
        return m_rigidSolver;
    }
    void setRigidSolver(std::shared_ptr<PBDSolver> rigidSolver)
    {
        m_rigidSolver = rigidSolver;
    }

    std::shared_ptr<SandInteractionForceSolver> getInteractionSolver() const
    {
        return m_interactSolver;
    }

    std::shared_ptr<PBDDensitySolver2D> getDensitySolver() const
    {
        return m_densitySolver;
    }

    void detectLandRigidContacts(PBDSolver*, Real);

    void setCallbackFunction(CallbackFun fun)
    {
        m_callback = fun;
    }

private:
    void _setRigidForceAsGravity();
    void _setRigidForceEmpty();

    //void _updateLandHeightField();

public:
    DEF_EMPTY_VAR(InteractionStepPerFrame, int, "Interaction steps per frame.");

    DEF_EMPTY_VAR(RigidStepPerInteraction, int, "Rigid body simulation step per interaction.");

    DEF_EMPTY_VAR(Contacts, DeviceDArray<ContactInfo<double>>, "Contact information");

    DEF_EMPTY_VAR(DetectThreshold, double, "Land Rigid Contact Detection threshold");

    DEF_EMPTY_VAR(BouyancyFactor, double, "Buoyancy Force Parameter");

    DEF_EMPTY_VAR(DragFactor, double, "Drag Force Paremeter");

    DEF_EMPTY_VAR(CHorizontal, double, "Horizontal Velocity Coupling Paremeter");

    DEF_EMPTY_VAR(CVertical, double, "Horizontal Velocity Coupling Paremeter");

    DEF_EMPTY_VAR(Cprobability, double, "Particle couple probability factor");

private:
    std::shared_ptr<PBDSandSolver> m_sandSolver;

    std::shared_ptr<PBDSolver> m_rigidSolver;

    std::shared_ptr<SandInteractionForceSolver> m_interactSolver;

    std::shared_ptr<PBDDensitySolver2D> m_densitySolver;

    std::shared_ptr<HeightFieldBodyDetector> m_landRigidContactDetector;

    //DeviceDArray<ContactInfo<double>> m_contacts;
    DeviceArray<PBDBodyInfo<double>> m_rigidBody;
    DeviceArray<PBDBodyInfo<double>> m_particleBody;
    //DeviceDArray<PBDJoint<double>> m_contactJoints;

    DeviceHeightField1d* m_landHeight;

    double m_gravity = 9.8;

    CallbackFun m_callback;

private:
    double m_totalTime  = 0.0;
    int    m_totalFrame = 0.0;
};
}  // namespace PhysIKA

#endif  // PARTICLESANDRIGIDINTERACTION_H