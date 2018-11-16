#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Framework/Framework/ModuleForce.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Kernel.h"
#include "Physika_Core/DataTypes.h"
#include "Attribute.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {
	template<typename TDataType>
	class ViscosityBase : public ForceModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ViscosityBase();
		~ViscosityBase() override {};
		
		bool execute() override;

// 		static ViscosityBase<TDataType>* Create(ViscosityBase<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new ViscosityBase(parent, deviceType);
// 		}


		bool connectPosition(std::shared_ptr<Field>& pos) { return connect(pos, m_position); }
		bool connectVelocity(std::shared_ptr<Field>& vel) { return connect(vel, m_velocity); }
		bool connectAttribute(std::shared_ptr<Field>& att) { return connect(att, m_attribute); }

		virtual bool connectDensity(std::shared_ptr<Field>& rho) { return connect(rho, m_density); }
		virtual bool connectRadius(std::shared_ptr<Field>& radius) { return connect(radius, m_radius); }
		virtual bool connectSamplingDistance(std::shared_ptr<Field>& dist) { return connect(dist, m_samplingDistance); }
		virtual bool connectNeighbor(std::shared_ptr<Field>& neighbor) { return connect(neighbor, m_neighbors); }


	private:
		DeviceBuffer<Coord>* m_oldVel;
		DeviceBuffer<Coord>* m_bufVel;

		HostVariablePtr<Real> m_viscosity;

		Slot<HostVariable<Real>>  m_radius;
		Slot<HostVariable<Real>>  m_samplingDistance;
		Slot<DeviceBuffer<Coord>> m_position;
		Slot<DeviceBuffer<Coord>> m_velocity;
		Slot<DeviceBuffer<Attribute>> m_attribute;
		Slot<DeviceBuffer<Real>>  m_density;
		Slot<DeviceBuffer<SPHNeighborList>> m_neighbors;
	};

#ifdef PRECISION_FLOAT
	template class ViscosityBase<DataType3f>;
#else
	template class ViscosityBase<DataType3d>;
#endif
}