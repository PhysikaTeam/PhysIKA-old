#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"

namespace Physika {

	template<typename TDataType>
	class SurfaceTension : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceTension(ParticleSystem<TDataType>* parent);
		~SurfaceTension() override {};
		
		bool execute() override;

		bool updateStates() override;

// 		static SurfaceTension* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new SurfaceTension(parent, deviceType);
// 		}

	private:
		ParticleSystem<TDataType>* m_parent;

		DeviceBuffer<float>* m_energy;
	};

	template class SurfaceTension<DataType3f>;
}