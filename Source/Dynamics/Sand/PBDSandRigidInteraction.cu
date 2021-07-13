#include "PBDSandRigidInteraction.h"

PhysIKA::PBDSandRigidInteraction::PBDSandRigidInteraction()
{
    //m_sandSolver = std::make_shared<PBDSandSolver>();
    //m_rigidSolver = std::make_shared<PBDSolver>();

    m_detector      = std::make_shared<PointMultiSDFContactDetector>();
    m_contactSolver = std::make_shared<PBDParticleBodyContactSolver>();
    m_densitySolver = std::make_shared<PBDDensitySolver2D>();
}

bool PhysIKA::PBDSandRigidInteraction::initialize()
{
    //if(m_rigidSolver)
    //	m_rigidSolver->initialize();

    //if(m_sandSolver)
    //	m_sandSolver->initialize();

    if (m_rigidSolver && m_sandSolver)
    {
        // Contact detector.
        m_sandSolver->needPosition3D(true);
        m_detector->m_contacts    = &m_contacts;
        m_detector->m_particlePos = &(m_sandSolver->getParticlePosition3D());
        m_detector->m_body        = &(m_rigidSolver->getGPUBody());

        // Contact solver.
        m_contactSolver->m_body         = &(m_rigidSolver->getGPUBody());
        m_contactSolver->m_particle     = &m_particleBody;
        m_contactSolver->m_contacts     = &m_contacts;
        m_contactSolver->m_joints       = &m_contactJoints;
        m_contactSolver->m_particleVel  = &(m_sandSolver->getParticleVelocity());
        m_contactSolver->m_particlePos  = &(m_sandSolver->getParticlePosition3D());
        m_contactSolver->m_particleMass = &(m_sandSolver->getParticleMass());

        // Sand density solver.
        m_densitySolver->m_particlePos   = &(m_sandSolver->getParticlePosition());
        m_densitySolver->m_particleVel   = &(m_sandSolver->getParticleVelocity());
        m_densitySolver->m_particleRho2D = &(m_sandSolver->getParticleRho2D());
        m_densitySolver->m_particleMass  = &(m_sandSolver->getParticleMass());
        //m_densitySolver->m_neighbor = &(m_sandSolver->getNeighborList());
        m_sandSolver->getNeighborField().connect(&(m_densitySolver->m_neighbor));

        m_densitySolver->m_smoothLength = m_sandSolver->getSmoothingLength();
        m_densitySolver->m_rho0         = m_sandSolver->getRho0();
    }
    return true;
}

void PhysIKA::PBDSandRigidInteraction::advance(Real dt)
{
    double subdt = dt / m_subStep;
    for (int i = 0; i < /*m_subStep*/ 1; ++i)
    {
        this->advectSubStep(subdt);
    }
}

void PhysIKA::PBDSandRigidInteraction::advectSubStep(Real dt)
{

    // Advect rigid.
    if (m_rigidSolver)
        m_rigidSolver->forwardSubStepGPU(dt);

    // Advect sand.
    if (m_sandSolver)
        m_sandSolver->forwardSubStep(dt);

    if (m_rigidSolver && m_sandSolver)
    {
        // Contact detection.
        m_detector->compute();

        // Do interaction constriant.
        m_contactSolver->updateParticleBodyInfo(1.0);
        m_contactSolver->buildJoints();
        m_contactSolver->forwardSubStep(dt, true);

        // Solver sand density constraint.
        m_densitySolver->forwardOneSubStep(dt);
    }
}
