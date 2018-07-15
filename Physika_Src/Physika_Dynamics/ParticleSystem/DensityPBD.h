#ifndef FRAMEWORK_DENSITYPBD_H
#define FRAMEWORK_DENSITYPBD_H

#include "Platform.h"
#include "Physika_Core/DataTypes.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "DensityConstraint.h"
#include "Kernel.h"

namespace Physika {

	template<typename TDataType>
	class DensityPBD : public DensityConstraint<TDataType>
	{
		DECLARE_CLASS_1(DensityPBD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityPBD();
		DensityPBD(ParticleSystem<TDataType>* parent);
		~DensityPBD() override {};
		
		bool execute() override;

		bool updateStates() override;

// 		static DensityPBD* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new DensityPBD(parent, deviceType);
// 		}

	protected:
		int m_maxIteration;
		ParticleSystem<TDataType>* m_parent;

		DeviceBuffer<Real>* m_lamda;
		DeviceBuffer<Coord>* m_deltaPos;
	};

#ifdef PRECISION_FLOAT
	template class DensityPBD<DataType3f>;
#else
 	template class DensityPBD<DataType3d>;
#endif
}

#endif