#pragma once
#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/Joint.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Core/Quaternion/quaternion.h"
#include "Framework/Topology/Frame.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Framework/Mapping/FrameToPointSet.h"
#include "Framework/Topology/TriangleSet.h"
#include "Rendering/RigidMeshRender.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Inertia.h"

#include <string>

namespace PhysIKA
{
	
	/*!
	*	\class	RigidBody
	*	\brief	Rigid body dynamics.
	*
	*	This class implements a simple rigid body.
	*
	*/
	template<typename TDataType>
	class RigidBody2 : public Node
	{
		DECLARE_CLASS_1(RigidBody2, TDataType)
	public:
		
		RigidBody2(std::string name = "default");
		virtual ~RigidBody2();

		virtual void loadShape(std::string filename);

		void addChildJoint(std::shared_ptr<Joint> child_joint);
		//void addChildJoint(Joint* child_joint);
		//void setParentJoint(std::shared_ptr<Joint> parent_joint);
		void setParentJoint(Joint* parent_joint);

		const ListPtr<Joint> & getChildJoints()
		{
			return m_child_joints;
		}

		Joint* getParentJoint()const { return m_parent_joint; }
		const ListPtr<Joint>& getChildJoints() const { return m_child_joints; }
		//ListPtr<Joint<TDataType>>& getChildJoints() { return m_child_joints; }

		const Inertia<float>& getI() { return m_I; }

		void setI(const Inertia<float>& I) 
		{
			m_I = I; 
			this->setMass(I.getMass());
		}

		// get all descendant nodes
		// virtual std::vector<std::shared_ptr<RigidBody2<TDataType>>> getAllNode();
		// virtual void getAllNode(std::vector<std::shared_ptr<RigidBody2<TDataType>>>& all_node);

		// get all descendant nodes in pair
		// pair: first: parent id in the vector; second: shared_ptr of node
		//virtual const std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& getAllParentidNodePair()const;
		//virtual void getAllParentidNodePair(std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& all_node);

		virtual void setGeometrySize(float sx, float sy, float sz)
		{
			m_sizex = sx;
			m_sizey = sy;
			m_sizez = sz;
		}

		Vector3f getGlobalR() const { return m_global_r; }
		Quaternion<float> getGlobalQ() const { return m_global_q; }

		void setGlobalR(const Vector3f& r) { m_global_r = r; }
		void setGlobalQ(const Quaternion<float>& q) { m_global_q = q; }

	
		void advance(Real dt) override;
		void updateTopology() override;

		int getId()const { return m_id; }
		void setId(int id) { m_id = id; }

	public:
		bool initialize() override;

		

	private:
		Joint* m_parent_joint;			// the parent joint
		ListPtr<Joint> m_child_joints;					// list of child joints

		int m_id=-1;

		// Physical property info
		Inertia<float> m_I;									// inertia

		// Geometric info
		float m_sizex = 1.0;
		float m_sizey = 1.0;
		float m_sizez = 1.0;
		double m_global_scale = 0.1;

		// state info
		Vector3f m_global_r;
		Quaternion<float> m_global_q;

		// others
		std::shared_ptr<TriangleSet<TDataType>> m_triSet;
		std::shared_ptr<Frame<TDataType>> m_frame;
		std::shared_ptr<RigidMeshRender> m_render;
		std::shared_ptr<FrameToPointSet<TDataType>> m_surfaceMapping;

	};



#ifdef PRECISION_FLOAT
	template class RigidBody2<DataType3f>;
	typedef std::shared_ptr<RigidBody2<DataType3f>> RigidBody2_ptr;

#else
	template class RigidBody2<DataType3d>;
	typedef std::shared_ptr<RigidBody2<DataType3d>> RigidBody2_ptr;

#endif
}