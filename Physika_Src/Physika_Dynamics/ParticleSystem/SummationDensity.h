#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "Kernel.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"

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

		bool initializeImpl() override;

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

		Slot<HostVarField<Real>>  m_mass;
		Slot<HostVarField<Real>>  m_radius;

		Slot<DeviceArrayField<Coord>> m_position;
		Slot<DeviceArrayField<Real>>  m_density;
		Slot<DeviceArrayField<SPHNeighborList>> m_neighbors;
	};

#ifdef PRECISION_FLOAT
	template class SummationDensity<DataType3f>;
#else
	template class SummationDensity<DataType3d>;
#endif
}