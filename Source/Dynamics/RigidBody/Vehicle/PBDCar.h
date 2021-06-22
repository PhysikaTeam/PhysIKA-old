#pragma once


#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"

//using namespace PhysIKA;

namespace PhysIKA
{
	class PBDCar :public Node
	{
	public:
		PBDCar() {}

		bool build();

		//virtual bool initialize() override;

		virtual void advance(Real dt);


		void forward(Real dt);

		void backward(Real dt);

		void goLeft(Real dt);

		void goRight(Real dt);

		//void setDt(Real dt);

		std::shared_ptr<RigidBody2<DataType3f>> getChassis() { return m_chassis; }
		std::shared_ptr<RigidBody2<DataType3f>> getWheels(int i) { return m_wheels[i]; }


		void updateForce(Real dt);


	private:
		void _updateWheelRotation(Real dt);

		Vector3f _getForwardDir();

		Vector3f _getRightDir();

		Vector3f _getUpDir();

		void _setRigidForceAsGravity();

		void _doVelConstraint(Real dt);

		Quaternionf _rotationToStandardLocal();

	public:
		Vector3f carPosition;
		Quaternion<float> carRotation;

		Vector3f wheelRelPosition[4];
		Quaternion<float> wheelRelRotation[4];

		Vector3f wheelupDirection;
		Vector3f wheelRightDirection;			// wheel right direction in car frame.


		// Visualization information.
		bool needVisualization = true;
		std::string chassisFile = "";
		std::string wheelFile[4] = { "","","","" };
		Vector3f chassisMeshScale;
		Vector3f wheelMeshScale[4];
		Vector3f chassisMeshTranslate;
		Vector3f wheelMeshTranslate[4];

		float chassisMass = 1.0;
		Vector3f chassisInertia;
		// adjust by HNU
		// C2397	从“double”转换到“float”需要收缩转换		
		float wheelMass[4] = { 0.1f,0.1f,0.1f,0.1f };
		Vector3f wheelInertia[4];

		float wheelRadius[4] = { 1.0,1.0,1.0,1.0 };

		float steeringLowerBound;
		float steeringUpperBound;

		Vector3f wheelLocalRight[2];		// wheel right direction in wheel local frame.

		Vector3f m_gravity = { 0,-9.8,0 };
	
		//std::shared_ptr<RigidBody2<DataType3f>> m_steeringRigid[2];

		std::shared_ptr<PBDSolver> m_rigidSolver;

		std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

		std::shared_ptr<RigidBody2<DataType3f>>m_steeringRigid[4];
		std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

		
		int chassisCollisionGroup = 1;
		int chassisCollisionMask = 1;
		int wheelCollisionGroup = 1;
		int wheelCollisionMask = 1;


		float forwardForceAcc;
		//float breakForceAcc;
		float steeringSpeed;

		float maxVel = 2.5;

		float linearDamping = 0;
		float angularDamping = 0;

		float suspensionLength = 0.05;
		float suspensionStrength = 1000000;

	private:
		Vector3f forwardForcePoint;
		Vector3f forwardDir;
		float forwardForce = 0;

		float breakForce = 0;


		float currentSteering = 0;

		bool m_accPressed = false;

		//float m_curSuspensionExt[0] 

	};

}