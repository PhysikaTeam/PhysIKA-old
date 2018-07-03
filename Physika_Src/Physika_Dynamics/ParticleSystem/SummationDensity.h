#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"

namespace Physika {

	template<typename TDataType>
	class SummationDensity : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

// 		static SummationDensity* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new SummationDensity(parent, deviceType);
// 		}

		SummationDensity(ParticleSystem<TDataType>* parent);

		~SummationDensity() override {};
		
		bool execute() override;

		bool updateStates() override;

		void SetCorrectFactor(Real factor) { m_factor = factor; }

	private:
		int m_maxIteration;
		Real m_factor;
		ParticleSystem<TDataType>* m_parent;
	};

	template class SummationDensity<DataType3f>;
}