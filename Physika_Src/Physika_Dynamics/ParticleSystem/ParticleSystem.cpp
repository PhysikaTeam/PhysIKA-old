#include "ParticleSystem.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleSystem, TDataType)

	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem(std::string name)
		: Node(name)
	{
		initField(&m_position, MechanicalState::position(), "Storing the particle positions!", false);
		initField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
		initField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

		m_pSet = std::make_shared<PointSet<TDataType>>();
		this->setTopologyModule(m_pSet);

		auto pts = m_pSet->getPoints();
		m_pSet->scale(0.05);
		m_pSet->translate(Coord(0.5, 0.2, 0.5));

		m_position.setElementCount(pts.size());
		m_velocity.setElementCount(pts.size());
		m_force.setElementCount(pts.size());

		Function1Pt::copy(m_position.getValue(), pts);
		m_velocity.getReference()->reset();

		m_pointsRender = std::make_shared<PointRenderModule>();
		m_pointsRender->setColor(Vector3f(0.2f, 0.6, 1.0f));
		this->addVisualModule(m_pointsRender);
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{
		
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::translate(Coord t)
	{

	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::updateTopology()
	{
		auto pts = m_pSet->getPoints();
		Function1Pt::copy(pts, getPosition()->getValue());
	}

	template<typename TDataType>
	std::shared_ptr<PointRenderModule> ParticleSystem<TDataType>::getRenderModule()
	{
		return m_pointsRender;
	}
}