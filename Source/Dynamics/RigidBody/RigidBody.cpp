#include "RigidBody.h"
#include "Framework/Topology/Frame.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Mapping/FrameToPointSet.h"
#include "IO/Surface_Mesh_IO/ObjFileLoader.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(RigidBody, TDataType)


	template<typename TDataType>
	PhysIKA::RigidBody<TDataType>::RigidBody(std::string name)
		: Node(name)
		, m_quaternion(Quaternion<Real>(Matrix::identityMatrix()))
	{
		attachField(&m_mass, MechanicalState::mass(), "Total mass of the rigid body!", false);
		attachField(&m_center, MechanicalState::position(), "Center of the rigid body!", false);
		attachField(&m_transVelocity, MechanicalState::velocity(), "Transitional velocity of the rigid body!", false);
		attachField(&m_angularVelocity, MechanicalState::angularVelocity(), "Angular velocity of the rigid body!", false);
		attachField(&m_force, MechanicalState::force(), "Transitional force of the rigid body!", false);
		attachField(&m_torque, MechanicalState::torque(), "Angular momentum of the rigid body!", false);
		attachField(&m_angularMass, MechanicalState::angularMass(), "Angular momentum", false);
		attachField(&m_rotation, MechanicalState::rotation(), "Orientation", false);

		Coord trans(0.5, 0.2, 0.5);

		m_mass.setValue(Real(1));
		m_center.setValue(trans);
		m_transVelocity.setValue(Coord(0));
		m_angularVelocity.setValue(Coord(0, 0, 0));
		m_force.setValue(Coord(0));
		m_torque.setValue(Coord(0));
		m_angularMass.setValue(Matrix::identityMatrix());
		m_rotation.setValue(Matrix::identityMatrix());

		m_frame = std::make_shared<Frame<TDataType>>();
		this->setTopologyModule(m_frame);

		m_frame->setCenter(trans);
		m_frame->setOrientation(m_quaternion.get3x3Matrix());

		//create a child node for surface rendering
		m_surfaceNode = this->createChild<Node>("Mesh");
		m_surfaceNode->setActive(false);
		m_surfaceNode->setControllable(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);
		triSet->scale(0.05);
		triSet->translate(trans);

		cuSynchronize();

		auto surfaceMapping = std::make_shared<FrameToPointSet<TDataType>>(m_frame, triSet);
		this->addTopologyMapping(surfaceMapping);
		cuSynchronize();
	}

	template<typename TDataType>
	RigidBody<TDataType>::~RigidBody()
	{
	}

	template<typename TDataType>
	bool RigidBody<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	void RigidBody<TDataType>::loadShape(std::string filename)
	{
		printf("surface\n");
		std::shared_ptr<TriangleSet<TDataType>> surface = TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule());
		surface->loadObjFile(filename);
	}

	template<typename TDataType>
	void RigidBody<TDataType>::setMass(Real mass)
	{
		m_mass.setValue(mass);
	}

	template<typename TDataType>
	void RigidBody<TDataType>::setCenter(Coord center)
	{
		m_center.setValue(center);
	}

	template<typename TDataType>
	void RigidBody<TDataType>::setVelocity(Coord vel)
	{
		m_transVelocity.setValue(vel);
	}

	template<typename TDataType>
	void RigidBody<TDataType>::advance(Real dt)
	{
		Real mass = m_mass.getValue();
		Coord center = m_center.getValue();
		Coord transVel = m_transVelocity.getValue();
		Coord angularVel = m_angularVelocity.getValue();
		Matrix angularMass = m_angularMass.getValue();

		Coord force = m_force.getValue();
		Coord forceMoment = m_torque.getValue();

		Matrix invMass = angularMass;
		angularVel += dt*(invMass*forceMoment);
		transVel += dt*force / mass + dt*Coord(0.0f, -9.8f, 0.0f);

		m_quaternion = m_quaternion + (0.5f * dt) * Quaternion<Real>(0, angularVel[0], angularVel[1], angularVel[2])*(m_quaternion);

		m_quaternion.normalize();
		center += transVel*dt;

		m_center.setValue(center);
		m_transVelocity.setValue(transVel);
		m_angularVelocity.setValue(angularVel);
		m_rotation.setValue(m_quaternion.get3x3Matrix());
	}

	template<typename TDataType>
	void RigidBody<TDataType>::updateTopology()
	{
		m_frame->setCenter(m_center.getValue());
		m_frame->setOrientation(m_quaternion.get3x3Matrix());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}


	//TODO:
	template<typename TDataType>
	void RigidBody<TDataType>::scale(Real t)
	{

	}

	template<typename TDataType>
	void RigidBody<TDataType>::translate(Coord t)
	{
		Coord center = m_center.getValue();
		m_center.setValue(center + t);

		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);
		//TypeInfo::CastPointerDown<PointSet<TDataType>>(m_collisionNode->getTopologyModule())->translate(t);
	}
}