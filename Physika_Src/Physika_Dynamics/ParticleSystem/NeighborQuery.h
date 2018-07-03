#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"
#include "GridHash.h"

namespace Physika {
	template<typename TDataType>
	class NeighborQuery : public Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborQuery(ParticleSystem<TDataType>* parent);
		~NeighborQuery() override {};
		
		bool execute() override;

// 		static NeighborQuery* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new NeighborQuery(parent, deviceType);
// 		}

	private:
		int m_maxIteration;
		ParticleSystem<TDataType>* m_parent;
		GridHash<TDataType> hash;
	};

	template class NeighborQuery<DataType3f>;
}