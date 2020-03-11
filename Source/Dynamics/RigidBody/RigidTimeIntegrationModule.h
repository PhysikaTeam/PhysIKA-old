#pragma once

#include "Framework/Framework/Module.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Dynamics/RigidBody/RigidState.h"

#include "Dynamics/RigidBody/ArticulatedBodyFDSolver.h"

#include<memory>
#include<vector>

namespace PhysIKA
{


	class RigidTimeIntegrationModule:public Module
	{
		DECLARE_CLASS(RigidTimeIntegrationModule)
	
	public:
		

	public:

		RigidTimeIntegrationModule();

		bool initialize() {};

		virtual void begin();

		virtual bool execute();

		virtual void end() {};

		//virtual void updateSystemState(double dt);

		//virtual void updateSystemState(const RigidState& s);
		
		void setDt(double dt) { m_dt = dt; }


		static void dydt(const SystemMotionState& s0, DSystemMotionState& ds);
		
	private:
		ArticulatedBodyFDSolver m_fd_solver;

		Vectornd<float> m_ddq;
		double m_dt = 0;

		double m_last_time = 0;
		bool m_time_init = false;
	};



}