#include "ParticleEmitter.h"

namespace PhysIKA
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter(std::string name)
		: ParticleSystem<TDataType>(name)
	{
	}

	template<typename TDataType>
	ParticleEmitter<TDataType>::~ParticleEmitter()
	{
		
	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::advance(Real dt)
	{
	}
}