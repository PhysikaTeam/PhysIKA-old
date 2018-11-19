#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/DataTypes.h"
#include "Kernel.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {

	template<typename TDataType>
	class SurfaceTension : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceTension();
		~SurfaceTension() override {};
		
		bool execute() override;

		bool updateStates() override;

// 		static SurfaceTension* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new SurfaceTension(parent, deviceType);
// 		}

	private:
		DeviceArrayField<Real>* m_energy;
	};

#ifdef PRECISION_FLOAT
	template class SurfaceTension<DataType3f>;
#else
	template class SurfaceTension<DataType3d>;
#endif
}