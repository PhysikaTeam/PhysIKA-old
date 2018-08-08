#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/DataTypes.h"
#include "INeighbors.h"
#include "Kernel.h"

namespace Physika {

	template<typename TDataType>
	class SummationDensity : public Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

// 		static SummationDensity* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new SummationDensity(parent, deviceType);
// 		}
		SummationDensity();

		~SummationDensity() override {};
		
		bool execute() override;

		bool initialize() override;

		bool updateStates() override;

		void SetCorrectFactor(Real factor) { m_factor = factor; }

		bool connectPosition(std::shared_ptr<Field>& pos) { return connect(pos, m_position); }
		bool connectMass(std::shared_ptr<Field>& mass) { return connect(mass, m_mass); }

		virtual bool connectDensity(std::shared_ptr<Field>& rho) { return connect(rho, m_density); }
		virtual bool connectRadius(std::shared_ptr<Field>& radius) { return connect(radius, m_radius); }
		virtual bool connectNeighbor(std::shared_ptr<Field>& neighbor) { return connect(neighbor, m_neighbors); }


	private:
		int m_maxIteration;
		Real m_factor;

		Slot<HostVariable<Real>>  m_mass;
		Slot<HostVariable<Real>>  m_radius;

		Slot<DeviceBuffer<Coord>> m_position;
		Slot<DeviceBuffer<Real>>  m_density;
		Slot<DeviceBuffer<SPHNeighborList>> m_neighbors;
	};

#ifdef PRECISION_FLOAT
	template class SummationDensity<DataType3f>;
#else
	template class SummationDensity<DataType3d>;
#endif
}