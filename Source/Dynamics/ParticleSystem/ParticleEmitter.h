#pragma once
#include "ParticleSystem.h"

namespace PhysIKA
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*
	*	
	*	
	*
	*/
	template<typename TDataType>
	class ParticleEmitter : public ParticleSystem<TDataType>
	{
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