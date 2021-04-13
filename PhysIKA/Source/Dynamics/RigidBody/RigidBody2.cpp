#include "RigidBody2.h"

#include "Core/Typedef.h"
#include <queue>
#include <random>
#include <ctime>
#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
//#include "Rendering/RigidMeshRender.h"
//#include "Rendering/SurfaceMeshRender.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(RigidBody2, TDataType)


	template<typename TDataType>
	PhysIKA::RigidBody2<TDataType>::RigidBody2(std::string name)
		: Node(name)//,m_I(6,6)
		//,m_X(6,6), m_v(6), m_a(6), m_external_f(6), m_r(6)
	{
		m_parent_joint = 0;
		m_triSet = 0;
		m_frame = 0;
		//m_render = 0;
		m_surfaceMapping = 0;


		//m_X.identity();
		//m_v.setZeros();
		//m_a.setZeros();

		//m_external_f.setZeros();
		
		//m_external_f(0) = 1;
		//m_external_f(1) = 1;
		//m_external_f(2) = 1;
		//m_external_f(3) = 1;
		//m_external_f(4) = -5;
		
	}

	template<typename TDataType>
	RigidBody2<TDataType>::~RigidBody2()
	{
		
	}

	template<typename TDataType>
	void RigidBody2<TDataType>::loadShape(std::string filename)
	{
		std::shared_ptr<TriangleSet<TDataType>> surface = m_triSet;// TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule());
		surface->loadObjFile(filename);
	}

	template<typename TDataType>
	void RigidBody2<TDataType>::addChildJoint(std::shared_ptr<Joint> child_joint)
	{
		m_child_joints.push_back(child_joint);
	}

	//template<typename TDataType>
	//void RigidBody2<TDataType>::addChildJoint(Joint* child_joint)
	//{
	//	m_child_joints.push_back(std::make_shared<Joint>(*child_joint));
	//}

	//template<typename TDataType>
	//void RigidBody2<TDataType>::setParentJoint(std::shared_ptr<Joint> parent_joint)
	//{
	//	m_parent_joint = parent_joint;
	//}

	template<typename TDataType>
	void RigidBody2<TDataType>::setParentJoint(Joint* parent_joint)
	{
		m_parent_joint = parent_joint;
	}
	

	//template<typename TDataType>
	//std::vector<std::shared_ptr<RigidBody2<TDataType>>> RigidBody2<TDataType>::getAllNode()
	//{
	//	std::vector<std::shared_ptr<RigidBody2<TDataType>>> all_node;
	//	getAllNode(all_node);

	//	return all_node;
	//}

	//template<typename TDataType>
	//void RigidBody2<TDataType>::getAllNode(std::vector<std::shared_ptr<RigidBody2<TDataType>>>& all_node)
	//{
	//	all_node.clear();

	//	Node* root = this;

	//	std::queue<std::shared_ptr<RigidBody2<TDataType>>> que;

	//	ListPtr<Node> root_childs = root->getChildren();
	//	for (auto iter = root_childs.begin(); iter != root_childs.end(); ++iter)
	//	{
	//		que.push(std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter));
	//	}

	//	while (!que.empty())
	//	{
	//		std::shared_ptr<RigidBody2<TDataType>> cur_node = que.front();
	//		all_node.push_back(cur_node);
	//		que.pop();

	//		ListPtr<Node> node_childs = cur_node->getChildren();
	//		for (auto iter = node_childs.begin(); iter != node_childs.end(); ++iter)
	//		{
	//			que.push(std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter));
	//		}
	//	}
	//}

	//template<typename TDataType>
	//const std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& RigidBody2<TDataType>::getAllParentidNodePair()const
	//{
	//	std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>  all_node;
	//	getAllParentidNodePair(all_node);

	//	return all_node;
	//}

	//template<typename TDataType>
	//void RigidBody2<TDataType>::getAllParentidNodePair(std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& all_node)
	//{
	//	all_node.clear();
	//	Node* root = this;

	//	// pair: first->parent id;  second->node pointer;
	//	std::queue<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>> que;

	//	ListPtr<Node> root_childs = root->getChildren();
	//	int cur_id = 0;
	//	for (auto iter = root_childs.begin(); iter != root_childs.end(); ++iter)
	//	{
	//		que.push(std::make_pair(-1, std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter)));
	//	}

	//	int i = -1;
	//	while (!que.empty())
	//	{
	//		++i;

	//		std::shared_ptr<RigidBody2<TDataType>> cur_node = que.front().second;
	//		all_node.push_back(que.front());
	//		que.pop();

	//		ListPtr<Node> node_childs = cur_node->getChildren();
	//		for (auto iter = node_childs.begin(); iter != node_childs.end(); ++iter)
	//		{
	//			que.push(std::make_pair(i, std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter)));
	//		}
	//	}
	//}

	//template<typename TDataType>
	//void RigidBody2<TDataType>::updateTopology()
	//{
	//	//m_frame->setCenter(m_global_r * m_global_scale);
	//	//m_frame->setOrientation(m_global_q.get3x3Matrix());
	//	//m_surfaceMapping->apply();

	//	//float rad;
	//	//Vector3f rotate_axis;
	//	//m_global_q.toRotationAxis(rad, rotate_axis);
	//	//Quaternion<float> rotation(rad, rotate_axis[0], rotate_axis[1], rotate_axis[2]);
	//	//m_render->setRotation(rotation);
	//	//m_render->setTranslatioin(m_global_r * m_global_scale);

	//	m_render->setTriangleRotation(m_global_q);
	//	m_render->setTriangleTranslation(m_global_r * m_global_scale);
	//}

	template<typename TDataType>
	bool RigidBody2<TDataType>::initialize()
	{
		
		m_triSet = std::make_shared<PhysIKA::TriangleSet<TDataType>>();
		m_triSet->loadObjFile("../../Media/standard/standard_cube.obj");
		
		m_triSet->translate(Vector3f(0.0, 0.0, 0.0));
		m_triSet->scale(Vector3f(m_sizex, m_sizey, m_sizez)* m_global_scale /2.0);
		//m_triSet->setIsRigid(true);
		this->setTopologyModule(m_triSet);

		//m_render = std::make_shared<RigidMeshRender>();
		
		//m_render->setColor(Vector3f(0.8, std::rand()%1000  / (double)1000, 0.8));
		//this->addVisualModule(m_render);

		m_frame = std::make_shared<Frame<TDataType>>();

		m_surfaceMapping = std::make_shared<FrameToPointSet<TDataType>>(m_frame, m_triSet);
		this->addTopologyMapping(m_surfaceMapping);

		this->setActive(true);
		this->setVisible(true);

		return true;
	}


	template<typename TDataType>
	void RigidBody2<TDataType>::advance(Real dt)
	{
		//// update simulation variables
		//auto time_integrator = this->getModule<RigidTimeIntegrationModule>();

		//time_integrator->begin();
		//time_integrator->setDt(dt);
		//time_integrator->execute();
		//time_integrator->end();

		m_parent_joint->update(dt);
	}
}