#pragma once
#include "DensityConstraint.h"

namespace Physika {

	template<typename TDataType>
	class DensitySimple : public DensityConstraint<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensitySimple(ParticleSystem<TDataType>* parent);
		~DensitySimple();

		bool execute() override;

		bool updateStates() override;

// 		static DensitySimple* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new DensitySimple(parent, deviceType);
// 		}

	private:
		DeviceBuffer<Coord>* m_dPos;
	};

	template class DensitySimple<DataType3f>;
}