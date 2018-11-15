#ifndef FRAMEWORK_DENSITYCONSTRAINT_H
#define FRAMEWORK_DENSITYCONSTRAINT_H

#include "Platform.h"
#include "Framework/ModuleConstraint.h"
#include "Physika_Core/DataTypes.h"
#include "Framework/FieldArray.h"
#include "Framework/FieldVar.h"
#include "Physika_Framework/Topology/INeighbors.h"
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
		~DensityConstraint() override {};
		
		bool updateStates() override;

// 		static DensityConstraint* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new DensityConstraint(parent, deviceType);
// 		}


		bool connectPosition(std::shared_ptr<Field>& pos) override { return connect(pos, m_position); }
		bool connectVelocity(std::shared_ptr<Field>& vel) override { return connect(vel, m_velocity); }
		
		virtual bool connectDensity(std::shared_ptr<Field>& rho)   { return connect(rho, m_density); }
		virtual bool connectRadius(std::shared_ptr<Field>& radius) { return connect(radius, m_radius); }
		virtual bool connectNeighbor(std::shared_ptr<Field>& neighbor) { return connect(neighbor, m_neighbors); }

	protected:
		Slot<HostVariable<Real>>  m_radius;
		Slot<DeviceBuffer<Coord>> m_position;
		Slot<DeviceBuffer<Coord>> m_velocity;
		Slot<DeviceBuffer<Real>>  m_density;
		Slot<DeviceBuffer<SPHNeighborList>> m_neighbors;
	};

#ifdef PRECISION_FLOAT
template class DensityConstraint<DataType3f>;
#else
template class DensityConstraint<DataType3d>;
#endif

}

#endif