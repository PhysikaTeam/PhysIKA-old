#pragma once
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{
	template <typename TDataType> class PointSet;
	class PointRenderModule;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleSystem : public Node
	{
		DECLARE_CLASS_1(ParticleSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleSystem(std::string name = "default");
		virtual ~ParticleSystem();

		void translate(Coord t);

		DeviceArrayField<Coord>* getPosition()
		{
			return &m_position;
		}

		DeviceArrayField<Coord>* getVelocity()
		{
			return &m_velocity;
		}

		DeviceArrayField<Coord>* getForce()
		{
			return &m_force;
		}

		DeviceArrayField<Vector3f>* getColor()
		{
			return &m_color;
		}

		void updateTopology() override;

		std::shared_ptr<PointRenderModule> getRenderModule();
	public:
		bool initialize() override;

	private:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Vector3f> m_color;
		DeviceArrayField<Coord> m_force;

		std::shared_ptr<PointSet<TDataType>> m_pSet;
		std::shared_ptr<PointRenderModule> m_pointsRender;
	};


#ifdef PRECISION_FLOAT
	template class ParticleSystem<DataType3f>;
#else
	template class ParticleSystem<DataType3d>;
#endif
}