#pragma once
#include "ParticleSystem.h"

namespace Physika
{
	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleElasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleElasticBody(std::string name = "default");
		virtual ~ParticleElasticBody();

		void advance(Real dt) override;

		void updateTopology() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		bool initialize() override;

		void loadSurface(std::string filename);

	private:
		std::shared_ptr<Node> m_surfaceNode;
	};

#ifdef PRECISION_FLOAT
	template class ParticleElasticBody<DataType3f>;
#else
	template class ParticleElasticBody<DataType3d>;
#endif
}