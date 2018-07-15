#ifndef FRAMEWORK_DENSITYCONSTRAINT_H
#define FRAMEWORK_DENSITYCONSTRAINT_H

#include "Platform.h"
#include "Framework/ModuleConstraint.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "ParticleSystem.h"
#include "Kernel.h"

namespace Physika {
	template<typename TDataType>
	class DensityConstraint : public ConstraintModule
	{
		DECLARE_CLASS_1(DensityConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityConstraint();
		DensityConstraint(ParticleSystem<TDataType>* parent);
		~DensityConstraint() override {};
		
		bool execute() override;

		bool updateStates() override;

		ParticleSystem<TDataType>* GetParent() { return m_parent; }
		void SetParent(ParticleSystem<TDataType>* p) { m_parent = p; }

// 		static DensityConstraint* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new DensityConstraint(parent, deviceType);
// 		}

	protected:
		ParticleSystem<TDataType>* m_parent;
	};
}

#ifdef PRECISION_FLOAT
template class DensityConstraint<DataType3f>;
#else
template class DensityConstraint<DataType3d>;
#endif

#endif