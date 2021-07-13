#pragma once

#ifndef _SANDSIMULATOR_H
#define _SANDSIMULATOR_H

#include "Framework/Framework/Node.h"
#include "SandSolverInterface.h"
#include "SandGrid.h"
//#include "Rendering/PointRenderModule.h"
#include "Framework/Topology/PointSet.h"
#include "Core/DataTypes.h"

#include <memory>

namespace PhysIKA {
class SandSimulator : public Node
{
public:
    SandSimulator();

    virtual ~SandSimulator();

    bool initialize() override;

    void updateTopology() override;

    void advance(Real dt) override;

    /**
        *@brief Set simulation solver of sand.
        */
    void setSandSolver(std::shared_ptr<SandSolverInterface> sandSolver);

    /**
        *@brief Set rendering particles sampler.
        */
    //void setRenderParticleSampler(std::shared_ptr<RenderParticleSampler> sampler);

    /**
        *@brief Set render particle sampler and particle point set for rendering preparation.
        */
    //void prepareRenderData(std::shared_ptr<RenderParticleSampler> sampler = std::make_shared<RenderParticleSampler>(),
    //    std::shared_ptr<PointSet<DataType3f>> particleSet = std::make_shared<PointSet<DataType3f>>());

    //SandGrid& getSandGrid() { return m_sandData; }

    //void setHeightFieldSample(bool useSample = true) { m_useHeightFieldSample = useSample; }

    void needForward(bool needforward)
    {
        m_needForward = needforward;
    }

public:
    // sand grid data.
    //SandGrid m_sandData;
    //SandGridInfo m_sandinfo;

    // sand simulation solver, such as: SSE solver.
    std::shared_ptr<SandSolverInterface> m_psandSolver;

    // Sampler will be used for the sampling of rendering particles.
    //std::shared_ptr<RenderParticleSampler> m_renderParticleSampler;

    // Topology module of sand.
    std::shared_ptr<PointSet<DataType3f>> m_renderParticleSet;

    //bool m_useHeightFieldSample = true;
    //int m_userParticleCount = 1;

    bool m_needForward = true;
};

}  // namespace PhysIKA
#endif  // _SANDSIMULATOR_H
