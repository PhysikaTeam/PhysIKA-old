#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Kernel.h"
#include "Physika_Framework/Topology/GridHash.h"

namespace Physika {
	template<typename TDataType>
	class NeighborQuery : public Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborQuery();
		~NeighborQuery() override {};
		
		bool execute() override;

		virtual bool connectPosition(std::shared_ptr<Field>& pos) { return connect(pos, m_position); }
		virtual bool connectNeighbor(std::shared_ptr<Field>& neighbor) { return connect(neighbor, m_neighbors); }
		virtual bool connectSamplingDistance(std::shared_ptr<Field>& sDist) { return connect(sDist, m_samplingDistance); }
		virtual bool connectSmoothingLength(std::shared_ptr<Field>& sLen) { return connect(sLen, m_smoothingLength); }

// 		static NeighborQuery* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new NeighborQuery(parent, deviceType);
// 		}

	private:
		int m_maxIteration;
		GridHash<TDataType> hash;

		Slot<HostVarField<Real>>  m_samplingDistance;
		Slot<HostVarField<Real>>  m_smoothingLength;
		Slot<DeviceArrayField<Coord>> m_position;
		Slot<DeviceArrayField<SPHNeighborList>> m_neighbors;
	};

#ifdef PRECISION_FLOAT
	template class NeighborQuery<DataType3f>;
#else
	template class NeighborQuery<DataType3d>;
#endif
}