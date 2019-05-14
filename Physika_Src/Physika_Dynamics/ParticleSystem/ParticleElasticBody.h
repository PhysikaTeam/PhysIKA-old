#pragma once
#include "ParticleSystem.h"

namespace Physika
{
	/*!
	*	\class	ParticleElasticBody
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
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

	private:

		

	};

#ifdef PRECISION_FLOAT
	template class ParticleElasticBody<DataType3f>;
#else
	template class ParticleElasticBody<DataType3d>;
#endif
}