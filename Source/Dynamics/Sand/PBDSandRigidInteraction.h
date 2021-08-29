#pragma once

#ifndef PBDSANDRIGIDINTERACTION_H
#define PBDSANDRIGIDINTERACTION_H

#include "Dynamics/Sand/PBDSandSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PointSDFContactDetector.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
class PBDSandRigidInteraction : public Node
{
public:
    PBDSandRigidInteraction();
    ~PBDSandRigidInteraction() {}

    virtual bool initialize() override;

    virtual void advance(Real dt) override;
    //void advect(Real dt);

    //void addSandChild()

    void advectSubStep(Real dt);

    std::shared_ptr<PointMultiSDFContactDetector> getContactDetector() const
    {
        return m_detector;
    }

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

    std::shared_ptr<PBDParticleBodyContactSolver> getParticleRigidContactSolver() const
    {
        return m_contactSolver;
    }
    std::shared_ptr<PBDDensitySolver2D> getDensitySolver() const
    {
        return m_densitySolver;
    }

private:
private:
    std::shared_ptr<PBDSandSolver> m_sandSolver;

    std::shared_ptr<PBDSolver> m_rigidSolver;

    std::shared_ptr<PointMultiSDFContactDetector> m_detector;

    std::shared_ptr<PBDParticleBodyContactSolver> m_contactSolver;

    std::shared_ptr<PBDDensitySolver2D> m_densitySolver;

    DeviceDArray<ContactInfo<double>> m_contacts;
    DeviceArray<PBDBodyInfo<double>>  m_rigidBody;
    DeviceArray<PBDBodyInfo<double>>  m_particleBody;
    DeviceDArray<PBDJoint<double>>    m_contactJoints;

    int m_subStep = 4;
};
}  // namespace PhysIKA

#endif  // PBDSANDRIGIDINTERACTION_H