#pragma once

#ifndef PHYSIKA_HEIGHTFIELDPBDINTERACTIONNODE_H
#define PHYSIKA_HEIGHTFIELDPBDINTERACTIONNODE_H

#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Framework/Topology/HeightField.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolverNode.h"
#include "Dynamics/RigidBody/ContactInfo.h"

#include "Dynamics/RigidBody/PBDRigid/BodyContactDetector.h"

#include <unordered_map>

//#include <b>
#include <vector>
#include <memory>

namespace PhysIKA {
class HeightFieldPBDInteractionNode : public PBDSolverNode
{
public:
    enum HFDETECTION
    {
        POINTVISE,
        FACEVISE
    };

    HeightFieldPBDInteractionNode()
    {
        detectionMethod = HFDETECTION::POINTVISE;
        m_solver        = std::make_shared<PBDSolver>();

        m_solver->setNarrowDetectionFun(
            std::bind(&HeightFieldPBDInteractionNode::contactDetection,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2));

        m_rigidContactDetector = std::make_shared<OrientedBodyDetector>();

        m_solver->setUseGPU(false);
    }

    ~HeightFieldPBDInteractionNode();

    virtual bool initialize() override;

    virtual void advance(Real dt) override;

    void setSize(int nx, int ny);
    void setSize(int nx, int ny, float dx, float dy);

    const DeviceHeightField1d& getHeightField() const
    {
        return m_heightField;
    }
    DeviceHeightField1d& getHeightField()
    {
        return m_heightField;
    }

    void setDetectionMethod(HFDETECTION method)
    {
        detectionMethod = method;
    }

    void setRigidDataDirty(bool datadirty = true)
    {
        m_rigidDataDirty = datadirty;
    }

    void contactDetection(PBDSolver* solver, Real dt);

    std::shared_ptr<OrientedBodyDetector> getRigidContactDetector()
    {
        return m_rigidContactDetector;
    }

private:
private:
    int m_nx = 0, m_ny = 0;

    DeviceDArray<ContactInfo<double>> m_contacts;
    DeviceHeightField1d               m_heightField;

    DeviceArray<int> nContactsi;
    int              m_nContacts = 0;

    bool m_rigidDataDirty = true;

    float mu                = 0.9;
    int   m_nPermanentJoint = 0;

    HFDETECTION detectionMethod;

    std::shared_ptr<OrientedBodyDetector> m_rigidContactDetector;
};
}  // namespace PhysIKA

#endif  // PHYSIKA_HEIGHTFIELDPBDINTERACTIONNODE_H