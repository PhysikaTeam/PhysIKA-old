#include "SandSimulator.h"

#include "Dynamics/Sand/SSEUtil.h"
#include "SSESandSolver.h"
#include <iostream>

namespace PhysIKA {
PhysIKA::SandSimulator::SandSimulator()
{
    // Set default sand simulation solver.
    m_psandSolver = std::make_shared<SSESandSolver>();

    this->setDt(0.02f);
}

PhysIKA::SandSimulator::~SandSimulator()
{
}

bool PhysIKA::SandSimulator::initialize()
{
    //assert(m_psandData);

    //m_renderModule = std::make_shared<PointRenderModule>();
    //this->addVisualModule(m_renderModule);

    //m_frame = std::make_shared<Frame<TDataType>>();
    //m_surfaceMapping = std::make_shared<FrameToPointSet<TDataType>>(m_frame, m_triSet);
    //this->addTopologyMapping(m_surfaceMapping);

    float* tmpa = new float[10];
    memset(tmpa, 0, sizeof(float) * 10);
    DeviceArray<double> devTmpa;
    devTmpa.resize(10);
    cudaMemcpy(devTmpa.begin(), tmpa, sizeof(float) * 10, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    delete[] tmpa;
    devTmpa.release();

    this->setActive(true);
    this->setVisible(true);

    //m_sandData.updateLandGridHeight();

    cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (this->m_psandSolver)
    {
        //m_sandData.getSandGridInfo(m_sandinfo);
        //m_psandSolver->setSandGridInfo(&m_sandinfo);
        m_psandSolver->initialize();
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    this->updateTopology();

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    Node::initialize();

    return true;
}

void PhysIKA::SandSimulator::updateTopology()
{
    //if (!m_renderParticleSet)
    //	return;
    ////assert(m_renderParticleSampler);
    //if (m_renderParticleSampler && m_useHeightFieldSample)
    //{
    //	if (m_renderParticleSet->getPointSize() != m_renderParticleSampler->sampleCount())
    //		m_renderParticleSet->setSize(m_renderParticleSampler->sampleCount());
    //	//m_sandData.updateSandGridHeight();
    //	m_renderParticleSampler->doSampling(m_renderParticleSet->getPoints().begin(),
    //		m_sandData.m_sandHeight, m_sandData.m_landHeight);
    //}
    //else
    //{
    //	m_psandSolver->updateUserParticle(m_renderParticleSet->getPoints());
    //}
}

void PhysIKA::SandSimulator::advance(Real dt)
{
    if (!m_needForward)
        return;

    float maxTimeStep = m_psandSolver->getMaxTimeStep();

    while (true)
    {
        float timeStep = dt > maxTimeStep ? maxTimeStep : dt;
        std::cout << "    Time step:  " << timeStep << std::endl;

        m_psandSolver->stepSimulation(timeStep);

        if (dt > maxTimeStep)
            dt -= maxTimeStep;
        else
            break;
    }

    //this->updateTopology();
}

void PhysIKA::SandSimulator::setSandSolver(std::shared_ptr<SandSolverInterface> sandSolver)
{
    m_psandSolver = sandSolver;
}

//void PhysIKA::SandSimulator::setRenderParticleSampler(std::shared_ptr<RenderParticleSampler> sampler)
//{
//	m_renderParticleSampler = sampler;
//}

//void PhysIKA::SandSimulator::prepareRenderData(std::shared_ptr<RenderParticleSampler> sampler, std::shared_ptr<PointSet<DataType3f>> particleSet)
//{

//	if (sampler && m_useHeightFieldSample)
//	{
//		m_renderParticleSampler = sampler;
//		int nx, ny;
//		m_sandData.getSize(nx, ny);
//		m_renderParticleSampler->Initalize(nx, ny, 2, 2, m_sandData.getGridLength());
//		m_renderParticleSampler->Generate();

//		m_userParticleCount = m_renderParticleSampler->sampleCount();
//	}
//	else
//	{

//	}

//	if (particleSet)
//	{
//		m_renderParticleSet = particleSet;

//		// Initialize topology module.
//		// For sand, use PointSet as its topology.
//		//m_renderParticleSet->setSize(m_renderParticleSampler->sampleCount());
//		this->setTopologyModule(m_renderParticleSet);
//	}
//}

}  // namespace PhysIKA