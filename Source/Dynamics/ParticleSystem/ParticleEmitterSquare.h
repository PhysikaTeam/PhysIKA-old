#pragma once
#include "ParticleSystem.h"

namespace PhysIKA
{
	/*!
	*	\class	ParticleFluid
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleEmitter : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleEmitter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();

		void advance(Real dt) override;
	private:
	};

#ifdef PRECISION_FLOAT
	template class ParticleEmitter<DataType3f>;
#else
	template class ParticleEmitter<DataType3d>;
#endif
}