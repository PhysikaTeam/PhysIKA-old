#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "DensityConstraint.h"
#include "Kernel.h"

namespace Physika {
	template<typename TDataType>
	class DensityProjective : public DensityConstraint<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityProjective(ParticleSystem<TDataType>* parent);
		~DensityProjective() override {};
		
		bool execute() override;

		bool updateStates() override;

		void SetWeight(Real w) { m_w = w; }

// 		static DensityProjective* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new DensityProjective(parent, deviceType);
// 		}

	protected:
		int m_maxIteration;
		ParticleSystem<TDataType>* m_parent;

		DeviceBuffer<Real>* m_lamda;
		DeviceBuffer<Coord>* m_deltaPos;
		DeviceBuffer<Coord>* m_posTmp;
		Real m_w;
	};

#ifdef PRECISION_FLOAT
	template class DensityProjective<DataType3f>;
#else
	template class DensityProjective<DataType3d>;
#endif
}