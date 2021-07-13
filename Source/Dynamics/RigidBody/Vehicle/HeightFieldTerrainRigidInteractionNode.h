#pragma once

#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Core/Array/Array2D.h"
#include "Framework/Topology/HeightField.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"

#include "Core/Utility/Reduction.h"
#include "Core/Utility/cuda_helper_math.h"

#include "FeathContactSolver.h"

#include "DantzigLCP.h"
#include <memory>

namespace PhysIKA {

struct TerrainRigidInteractionInfo
{
    Real surfaceThickness;
    Real elasticModulus;

    Real damping;
};

class HeightFieldTerrainRigidInteractionNode : public Node
{
public:
    enum HFDETECTION
    {
        POINTVISE,
        FACEVISE
    };

    HeightFieldTerrainRigidInteractionNode()
    {
        detectionMethod = HFDETECTION::POINTVISE;
    }

    ~HeightFieldTerrainRigidInteractionNode();

    bool initialize() override;

    void setRigidBodySystem(std::shared_ptr<RigidBodyRoot<DataType3f>> rigidsys)
    {
        m_rigidSystem = rigidsys;
    }

    virtual void advance(Real dt) override;

    void setSize(int nx, int ny);
    void setSize(int nx, int ny, float dx, float dy);

    const DeviceGrid2Df& getHeightField() const
    {
        return m_heightField;
    }
    DeviceGrid2Df& getHeightField()
    {
        return m_heightField;
    }

    void setTerrainInfo(TerrainRigidInteractionInfo info)
    {
        terrainInfo = info;
    }

    void setDetectionMethod(HFDETECTION method)
    {
        detectionMethod = method;
    }

private:
    bool _calcualteSingleContactForce(std::shared_ptr<RigidBody2<DataType3f>> prigid, Vector3f& force, Vector3f& torque);

    bool _test(Real dt);

private:
    std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

    int m_nx = 0, m_ny = 0;
    //DeviceArray2D<float> m_heightField;

    //HeightField<DataType3f> m_heightField;

    DeviceGrid2Df m_heightField;

    DeviceArray<Vector3f> m_interactForce;
    DeviceArray<Vector3f> m_interactTorque;

    std::shared_ptr<Reduction<Vector3f>> m_forceSummator;

    //std::shared_ptr<Reduction<ContactPointX>> m_maxDepthFinder;

    TerrainRigidInteractionInfo terrainInfo;

    DantzigScratchMemory m_dantzigscratch;
    DantzigInputMemory   m_dantzigInput;

    DeviceArray<ContactPointX> m_depthScratch;

    bool  solveFriction = true;
    float mu            = 0.9;

    HFDETECTION detectionMethod;
};

}  // namespace PhysIKA
