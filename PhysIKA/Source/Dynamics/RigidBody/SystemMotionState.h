#pragma once

#include "Framework/Framework/Module.h"

#include "Core/Vector/vector_nd.h"

#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/Transform3d.h"
#include "SpatialVector.h"

#include<memory>

namespace PhysIKA
{
	struct DSystemMotionState
	{
	public:

		void setRigidNum(int n)
		{
			m_rel_r.resize(n);
			m_rel_q.resize(n);
			m_v.resize(n);
		}

		void setDof(int n)
		{
			m_dq.resize(n);
		}

	public:
		std::vector<Vector3f> m_rel_r;						// relative position
		std::vector<Quaternion<float>> m_rel_q;				// relative rotation

		std::vector<SpatialVector<float>> m_v;				// Relative spatial velocities


		Vectornd<float> m_dq;
	};

	struct SystemMotionState//:public State
	{
	public:
		SystemMotionState(Node* root = 0) :m_root(root)
		{
			//build();
		}

		void setRoot(Node * root) { m_root = root; }


		//void build();
		SystemMotionState& addDs(const DSystemMotionState& ds, double dt);

		void updateGlobalInfo();

		void setRigidNum(int n)
		{

			m_rel_r.resize(n);
			m_rel_q.resize(n);
			m_global_r.resize(n);
			m_global_q.resize(n);
			m_X.resize(n);
			m_v.resize(n);
		}

		void setNum(int n, int dof)
		{
			m_rel_r.resize(n);
			m_rel_q.resize(n);
			m_global_r.resize(n);
			m_global_q.resize(n);
			m_X.resize(n);
			m_v.resize(n);

			m_dq.resize(dof);
		}

		//static void dydt(const SystemMotionState& s0, SystemMotionState& ds);


	private:



	public:
		//RigidBodyRoot<DataType3f>* m_root = 0;
		Node* m_root = 0;

		// ---------x
		std::vector<Vector3f> m_rel_r;
		std::vector<Quaternion<float>> m_rel_q;

		std::vector<Vector3f> m_global_r;					// global positions of rigid bodies
		std::vector<Quaternion<float>> m_global_q;			// global rotations of rigid bodies

		std::vector<Transform3d<float>> m_X;				// Transformations from parent nodes to child nodes


		// -------- v
		std::vector<SpatialVector<float>> m_v;				// Relative spatial velocities, successor frame

		// General velocities in joint space.
		// For eanch joint, its dof can be 0-6. 
		// It will be inefficient to use a Vectornd, as we need to allocate dynamic memory for eanch Vectornd.
		Vectornd<float> m_dq;								// General velocities in joint space


	};



	//class RigidSystemFor
}