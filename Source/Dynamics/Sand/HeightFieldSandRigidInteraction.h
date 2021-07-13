#pragma once

#ifndef HEIGHTFIELDSANDRIGIDINTERACTION_H
#define HEIGHTFIELDSANDRIGIDINTERACTION_H

#include "Dynamics/Sand/SSESandSolver.h"
//#include "Dynamics/Sand/PBDSandSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Dynamics/RigidBody/PBDRigid/PointSDFContactDetector.h"
#include "Framework/Framework/Node.h"
#include "Dynamics/Sand/SandInteractionForceSolver.h"

#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Dynamics/Sand/HeightFieldDensitySolver.h"

#include "Dynamics/RigidBody/PBDRigid/BodyContactDetector.h"

namespace PhysIKA {
class HeightFieldSandRigidInteraction : public Node
{
public:
    HeightFieldSandRigidInteraction();
    ~HeightFieldSandRigidInteraction() {}

    virtual bool initialize() override;

    virtual void advance(Real dt) override;
    //void advect(Real dt);

    //void addSandChild()

    void advectSubStep(Real dt);

    void setSandGrid(DeviceHeightField1d& sandHeight, DeviceHeightField1d& landHeight);

    std::shared_ptr<SSESandSolver> getSandSolver() const
    {
        return m_sandSolver;
    }
    void setSandSolver(std::shared_ptr<SSESandSolver> sandSolver)
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

    //std::shared_ptr<HeightFieldDensitySolver> getDensitySolver()const { return m_densitySolver; }

    void detectLandRigidContacts(PBDSolver*, Real);

public:
    DEF_EMPTY_VAR(Contacts, DeviceDArray<ContactInfo<double>>, "Contact information");

    DEF_EMPTY_VAR(DetectThreshold, double, "Land Rigid Contact Detection threshold");

    DEF_EMPTY_VAR(BouyancyFactor, double, "Buoyancy Force Parameter");

    DEF_EMPTY_VAR(DragFactor, double, "Drag Force Paremeter");

    DEF_EMPTY_VAR(CHorizontal, double, "Horizontal Velocity Coupling Paremeter");

    DEF_EMPTY_VAR(CVertical, double, "Horizontal Velocity Coupling Paremeter");

private:
    void _setRigidForceAsGravity();
    void _setRigidForceEmpty();

    void _updateSandHeightField();

    void _updateGridParticleInfo(int i);

    void _computeBoundingGrid(int& minGx, int& minGz, int& maxGx, int& maxGz, float radius, const Vector3f& center);

public:
    int m_subStep      = 1;
    int m_subRigidStep = 20;

    double m_gravity = 9.8;

    int m_boundingBlockMargin = 3;

private:
    std::shared_ptr<SSESandSolver> m_sandSolver;

    std::shared_ptr<PBDSolver> m_rigidSolver;

    std::shared_ptr<SandInteractionForceSolver> m_interactSolver;

    std::shared_ptr<HeightFieldBodyDetector> m_landRigidContactDetector;

    //std::shared_ptr< PBDDensitySolver2D> m_densitySolver;
    //std::shared_ptr<HeightFieldDensitySolver> m_densitySolver;

    DeviceDArray<Vector3d> m_gridParticle;
    DeviceDArray<Vector3d> m_gridVel;
    DeviceDArray<double>   m_gridMass;

    SandGridInfo*        m_psandInfo  = 0;
    DeviceHeightField1d* m_sandHeight = 0;
    DeviceHeightField1d* m_landHeight = 0;

    int m_minGi  = 0;
    int m_minGj  = 0;
    int m_sizeGi = 0;
    int m_sizeGj = 0;

    // debug
    double m_totalTime  = 0.0;
    int    m_totalFrame = 0;
};
}  // namespace PhysIKA

#endif  // HEIGHTFIELDSANDRIGIDINTERACTION_H