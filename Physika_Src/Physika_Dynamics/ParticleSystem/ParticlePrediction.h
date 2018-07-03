#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"

namespace Physika {
	template<typename TDataType>
	class ParticlePrediction : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticlePrediction(ParticleSystem<TDataType>* parent);
		~ParticlePrediction() override {};
		
		bool execute() override;

		void PredictPosition(float dt);
		void PredictVelocity(float dt);

		void CorrectPosition(float dt);

		bool updateStates() override;

	private:
		ParticleSystem<TDataType>* m_parent;
	};

	template class ParticlePrediction<DataType3f>;

// 	template class ParticlePrediction<DataType3d>;
}