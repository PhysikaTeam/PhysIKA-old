#include "ParticleSystem.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleSystem, TDataType)

	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem(std::string name)
		: Node(name)
	{
		attachField(&m_position, MechanicalState::position(), "Storing the particle positions!", false);
		attachField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
		attachField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

		m_pSet = std::make_shared<PointSet<TDataType>>();
		this->setTopologyModule(m_pSet);

		m_pointsRender = std::make_shared<PointRenderModule>();
		//m_pointsRender->setColor(Vector3f(0.2f, 0.6, 1.0f));
//		m_pointsRender->setColorArray(m_color);
		this->addVisualModule(m_pointsRender);
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{
		
	}


	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(std::string filename)
	{
		m_pSet->loadObjFile(filename);
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::translate(Coord t)
	{
		m_pSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		return true;
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::initialize()
	{
		return Node::initialize();
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::updateTopology()
	{
		auto pts = m_pSet->getPoints();
		Function1Pt::copy(pts, getPosition()->getValue());
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::resetStatus()
	{
		auto pts = m_pSet->getPoints();

		m_position.setElementCount(pts.size());
		m_velocity.setElementCount(pts.size());
		m_force.setElementCount(pts.size());

		Function1Pt::copy(m_position.getValue(), pts);
		m_velocity.getReference()->reset();

		return Node::resetStatus();
	}

	template<typename TDataType>
	std::shared_ptr<PointRenderModule> ParticleSystem<TDataType>::getRenderModule()
	{
		return m_pointsRender;
	}
}