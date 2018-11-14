#include "RigidBody.h"
#include "Framework/Node.h"
#include "Topology/PointSet.h"
#include "Framework/MechanicalState.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"


namespace Physika 
{
	IMPLEMENT_CLASS_1(RigidBody, TDataType)

	template<typename TDataType>
	RigidBody<TDataType>::RigidBody()
		: NumericalModel()
		, m_quaternion(Quaternion<Real>(Matrix::identityMatrix()))
	{
		
	}

	template<typename TDataType>
	bool RigidBody<TDataType>::initializeImpl()
	{
		this->NumericalModel::initializeImpl();

		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		PointSet<TDataType>* pSet = dynamic_cast<PointSet<TDataType>*>(parent->getTopologyModule().get());
		if (pSet == NULL)
		{
			Log::sendMessage(Log::Error, "The topology module is not supported!");
			return false;
		}

		if (!pSet->isInitialized())
		{
			pSet->initialize();
		}

		auto points = pSet->getPoints();
		HostArray<Coord> vertex;
		vertex.resize(points->size());
		Function1Pt::Copy(vertex, *points);
		Coord cenPos(0);
		for (size_t i = 0; i < vertex.size(); i++)
		{
			cenPos += vertex[i];
		}
		cenPos /= vertex.size();
		vertex.release();

		auto mstate = parent->getMechanicalState();
		mstate->setMaterialType(MechanicalState::RIGIDBODY);

		HostVariable<Real>::createField(mstate.get(), MechanicalState::mass(), "Mass", Real(1));
		HostVariable<Matrix>::createField(mstate.get(), MechanicalState::angularMass(), "Angular momentum", Matrix::identityMatrix());
		HostVariable<Matrix>::createField(mstate.get(), MechanicalState::rotation(), "Orientation", m_quaternion.get3x3Matrix());
		HostVariable<Coord>::createField(mstate.get(), MechanicalState::init_position(), "Initial position", cenPos);
		auto center = HostVariable<Coord>::createField(mstate.get(), MechanicalState::position(), "Mass center", cenPos);
		auto vel = HostVariable<Coord>::createField(mstate.get(), MechanicalState::velocity(), "Translational velocity", Coord(0, 0, 0));
		auto angularVel = HostVariable<RotateCoord>::createField(mstate.get(), MechanicalState::angularVelocity(), "Angular velocity", RotateCoord(0, 0, 0));
		HostVariable<Coord>::createField(mstate.get(), MechanicalState::force(), "Force", Coord(0));
		HostVariable<RotateCoord>::createField(mstate.get(), MechanicalState::forceMoment(), "Force moment", RotateCoord(0));

		m_mapping = std::make_shared<RigidToPoints<TDataType>>();
		m_mapping->initialize(Rigid(cenPos, m_quaternion), *(pSet->getPoints()));
	}

	template<typename TDataType>
	void RigidBody<TDataType>::step(Real dt)
	{
		auto mstate = getParent()->getMechanicalState();

		mstate->resetForce();

		auto forceModules = getParent()->getForceModuleList();
		for (std::list<std::shared_ptr<ForceModule>>::iterator iter = forceModules.begin(); iter != forceModules.end(); iter++)
		{
			(*iter)->applyForce();
		}

		auto massField = mstate->getField<HostVariable<Real>>(MechanicalState::mass());
		auto rotationField = mstate->getField<HostVariable<Matrix>>(MechanicalState::rotation());
		auto angularMassField = mstate->getField<HostVariable<Matrix>>(MechanicalState::angularMass());
		auto posField = mstate->getField<HostVariable<Coord>>(MechanicalState::position());
		auto transVelField = mstate->getField<HostVariable<Coord>>(MechanicalState::velocity());
		auto angularVelField = mstate->getField<HostVariable<RotateCoord>>(MechanicalState::angularVelocity());

		Real mass = massField->getValue();
		Coord center = posField->getValue();
		Coord transVel = transVelField->getValue();
		Coord angularVel = angularVelField->getValue();
		Matrix angularMass = angularMassField->getValue();

		Coord force = mstate->getField<HostVariable<Coord>>(MechanicalState::force())->getValue();
		Coord forceMoment = mstate->getField<HostVariable<Coord>>(MechanicalState::forceMoment())->getValue();

		Matrix invMass = angularMass;
		angularVel += dt*(invMass*forceMoment);
		transVel += dt*force / mass;

		m_quaternion = m_quaternion + (0.5f * dt) * Quaternion<Real>(0, angularVel[0], angularVel[1], angularVel[2])*m_quaternion;
		
		m_quaternion.normalize();
		m_displacement = transVel*dt;

		center += m_displacement;

		posField->setValue(center);
		transVelField->setValue(transVel);
		angularVelField->setValue(angularVel);
		rotationField->setValue(m_quaternion.get3x3Matrix());

		auto constraintModules = getParent()->getConstraintModuleList();
		for (std::list<std::shared_ptr<ConstraintModule>>::iterator iter = constraintModules.begin(); iter != constraintModules.end(); iter++)
		{
			(*iter)->constrain();
		}
	}

	template<typename TDataType>
	void RigidBody<TDataType>::updateTopology()
	{
		auto mstate = getParent()->getMechanicalState();
		auto center = mstate->getField<HostVariable<Coord>>(MechanicalState::position())->getValue();

		auto pSet = TypeInfo::CastPointerDown<PointSet<TDataType>>(getParent()->getTopologyModule());
		auto points = pSet->getPoints();

		m_mapping->applyTransform(Rigid(center, m_quaternion), *points);
	}
}