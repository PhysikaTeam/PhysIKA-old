#include "ParticleEmitter.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleEmitter, TDataType)

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