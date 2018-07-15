#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"

namespace Physika {
	template<typename TDataType>
	class ViscosityBase : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ViscosityBase(ParticleSystem<TDataType>* parent);
		~ViscosityBase() override {};
		
		bool execute() override;

		bool updateStates() override;

// 		static ViscosityBase<TDataType>* Create(ViscosityBase<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new ViscosityBase(parent, deviceType);
// 		}

	private:
		ParticleSystem<TDataType>* m_parent;

		DeviceBuffer<Coord>* m_oldVel;
		DeviceBuffer<Coord>* m_bufVel;
	};

#ifdef PRECISION_FLOAT
	template class ViscosityBase<DataType3f>;
#else
	template class ViscosityBase<DataType3d>;
#endif
}