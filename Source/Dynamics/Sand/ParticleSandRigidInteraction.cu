#include "Dynamics/Sand/ParticleSandRigidInteraction.h"

#include "Dynamics/Sand/SSEUtil.h"

#include "Core/Utility/CTimer.h"

namespace PhysIKA
{

	ParticleSandRigidInteraction::ParticleSandRigidInteraction()
	{
		m_interactSolver = std::make_shared< SandInteractionForceSolver>();
		m_landRigidContactDetector = std::make_shared<HeightFieldBodyDetector>();

		m_densitySolver = std::make_shared<PBDDensitySolver2D>();

		this->var_InteractionStepPerFrame.setValue(1);
		this->var_RigidStepPerInteraction.setValue(20);

		this->var_BouyancyFactor.setValue(300);
		this->var_DragFactor.setValue(1.0);
		this->var_CHorizontal.setValue(1.0);
		this->var_CVertical.setValue(1.0);
		this->var_Cprobability.setValue(100000.0);
	}

	bool ParticleSandRigidInteraction::initialize()
	{
		//if(m_rigidSolver)
		//	m_rigidSolver->initialize();

		//if(m_sandSolver)
		//	m_sandSolver->initialize();

		
		if (m_sandSolver)
		{
			m_landHeight = &(m_sandSolver->getLand());
		}

		if (m_rigidSolver && m_sandSolver)
		{
			m_sandSolver->needPosition3D(true);


			// Interaction solver.
			m_interactSolver->m_body = &(m_rigidSolver->getGPUBody());
			m_interactSolver->m_hostBody = m_rigidSolver->getCPUBody().size()>0? 
				&(m_rigidSolver->getCPUBody()[0]) : 0;
			m_interactSolver->m_particlePos = &(m_sandSolver->getParticlePosition3D());
			m_interactSolver->m_particleMass = &(m_sandSolver->getParticleMass());
			m_interactSolver->m_particleVel = &(m_sandSolver->getParticleVelocity());
			m_interactSolver->m_land = &(m_sandSolver->getLand());
			m_interactSolver->m_beAddForceUpdate = false;
			m_interactSolver->m_useStickParticleVelUpdate = false;
			m_interactSolver->m_CsHorizon = this->var_CHorizontal.getValue();
			m_interactSolver->m_CsVertical = this->var_CVertical.getValue();
			m_interactSolver->m_Cprob = this->var_Cprobability.getValue();
			m_interactSolver->m_Cdrag = this->var_DragFactor.getValue();
			m_interactSolver->m_buoyancyFactor = this->var_BouyancyFactor.getValue();
			m_interactSolver->m_e = 0.0;


			m_interactSolver->m_smoothLength = m_sandSolver->getSmoothingLength();
			//m_interactSolver->m_neighbor = &(m_sandSolver->getNeighborList());
			m_sandSolver->getNeighborField().connect(&(m_interactSolver->m_neighbor));

			// Sand density solver.
			m_densitySolver->m_particlePos = &(m_sandSolver->getParticlePosition());
			m_densitySolver->m_particleVel = &(m_sandSolver->getParticleVelocity());
			m_densitySolver->m_particleRho2D = &(m_sandSolver->getParticleRho2D());
			m_densitySolver->m_particleMass = &(m_sandSolver->getParticleMass());
			//m_densitySolver->m_neighbor = &(m_sandSolver->getNeighborList());
			m_sandSolver->getNeighborField().connect(&(m_densitySolver->m_neighbor));
			
			m_densitySolver->m_smoothLength = m_sandSolver->getSmoothingLength();
			m_densitySolver->m_rho0 = m_sandSolver->getRho0();
			m_densitySolver->minh = 0.05;
		}


		// Initialize contact detector
		this->var_Contacts.setValue(DeviceDArray< ContactInfo<double>>());
		this->var_DetectThreshold.setValue(0.0);
		if (m_landRigidContactDetector)
		{
			this->var_DetectThreshold.connect(m_landRigidContactDetector->varThreshold());
			this->var_Contacts.connect(m_landRigidContactDetector->varContacts());
			m_landRigidContactDetector->m_land = this->m_landHeight;

			if (m_rigidSolver)
			{
				// Init collision body ptr.
				auto& rigids = m_rigidSolver->getRigidBodys();
				for (auto prigid : rigids)
				{
					m_landRigidContactDetector->addCollidableObject(prigid);
				}

				// Set contact detection function.
				m_rigidSolver->setNarrowDetectionFun(std::bind(&ParticleSandRigidInteraction::detectLandRigidContacts,
					this, std::placeholders::_1, std::placeholders::_2));
			}


		}


		return true;
	}

	void ParticleSandRigidInteraction::advance(Real dt)
	{
		if (m_callback)
		{
			m_callback(this, dt);
		}

		CTimer timer;
		timer.start();

		int iterStep = var_InteractionStepPerFrame.getValue();
		double subdt = dt / iterStep;
		for (int i = 0; i < iterStep; ++i)
		{
			this->advectSubStep(subdt);
		}

		timer.stop();

		double curElaTime = timer.getElapsedTime();
		m_totalTime += curElaTime;
		m_totalFrame += 1;
		printf("  Cur elapsed time: %lf.  Average elapsed time: %lf. \n", curElaTime, m_totalTime / m_totalFrame);
	}

	void ParticleSandRigidInteraction::advectSubStep(Real dt)
	{
		// debug 
		CTimer timer;

		//// Advect sand.
		//if (m_sandSolver)
		//{
		//	timer.start();

		//	m_sandSolver->velocityUpdate(dt);

		//	timer.stop();
		//	printf("Sand velocity update time:   %lf\n", (double)(timer.getElapsedTime()));
		//}




		if (m_rigidSolver && m_sandSolver)
		{
			//timer.start();

			m_interactSolver->setPreBodyInfo();

			//timer.stop();
			//printf("Interactor save previous info time:   %lf\n", (double)(timer.getElapsedTime()));
		}

		if (m_rigidSolver)
		{
			//timer.start();

			int rigidStep = var_RigidStepPerInteraction.getValue();
			Real rdt = dt / rigidStep;
			for (int i = 0; i < rigidStep; ++i)
			{
				m_rigidSolver->forwardSubStepGPU2(rdt);
			}
			m_rigidSolver->updateGPUToCPUBody();

			//timer.stop();
			//printf("Rigid solve time:   %lf\n", (double)(timer.getElapsedTime()));

		}

		if (m_rigidSolver && m_sandSolver)
			//if(false)
		{
			//this->_updateSandHeightField();

			//timer.start();

			m_interactSolver->updateBodyAverageVel(dt);

			int nrigid = m_rigidSolver->getRigidBodys().size();
			for (int i = 0; i < nrigid; ++i)
			{
				if (m_interactSolver->collisionValid(m_rigidSolver->getRigidBodys()[i]))
				{
					//// Update info.
					//this->_updateGridParticleInfo(i);

					// Solve interaction force.
					m_interactSolver->computeSingleBody(i, dt);

					//// Solver sand density constraint.
					//m_sandSolver->applyVelocityChange(dt, m_minGi, m_minGj, m_sizeGi, m_sizeGj);
				}
			}

			// Solver sand density constraint.
			m_densitySolver->forwardOneSubStep(dt);

			//timer.stop();
			//printf("Interaction time:   %lf\n", (double)(timer.getElapsedTime()));
		}

		if (m_rigidSolver)
		{
			// Synchronize rigid body info.
			m_rigidSolver->synFromBodiedToRigid();
		}

		// Advect sand.
		if (m_sandSolver)
		{
			//timer.start();

			m_sandSolver->velocityUpdate(dt);

			//timer.stop();
			//printf("Sand velocity update time:   %lf\n", (double)(timer.getElapsedTime()));
		}

		if (m_sandSolver)
		{
			//timer.start();

			m_sandSolver->positionUpdate(dt);
			//timer.stop();
			//printf("Sand pos update time:   %lf\n\n", (double)(timer.getElapsedTime()));


			//timer.start();

			m_sandSolver->infoUpdate(dt);

			//timer.stop();
			//printf("Sand info update time:   %lf\n\n", (double)(timer.getElapsedTime()));
		}

	}


	void ParticleSandRigidInteraction::detectLandRigidContacts(PBDSolver *solver, Real dt)
	{
		if (m_landRigidContactDetector)
		{
			m_landRigidContactDetector->doCollision();

			auto& contactArr = m_landRigidContactDetector->varContacts()->getValue();
			m_rigidSolver->setContactJoints(contactArr,
				contactArr.size());
		}
	}

	void ParticleSandRigidInteraction::_setRigidForceAsGravity()
	{
		if (!m_rigidSolver) return;

		auto& rigids = m_rigidSolver->getRigidBodys();
		for (auto prigid : rigids)
		{
			prigid->setExternalForce(Vector3f(0, -m_gravity * prigid->getI().getMass(), 0));
			prigid->setExternalTorque(Vector3f(0, 0, 0));
		}

	}

	void ParticleSandRigidInteraction::_setRigidForceEmpty()
	{
		if (!m_rigidSolver) return;

		auto& rigids = m_rigidSolver->getRigidBodys();
		for (auto prigid : rigids)
		{
			prigid->setExternalForce(Vector3f(0, 0, 0));
			prigid->setExternalTorque(Vector3f(0, 0, 0));
		}
	}



}