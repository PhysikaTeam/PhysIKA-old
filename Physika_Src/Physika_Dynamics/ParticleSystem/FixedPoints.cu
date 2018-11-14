#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Framework/FieldVar.h"
#include "Framework/MechanicalState.h"
#include "Framework/Node.h"
#include "FixedPoints.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(FixedPoints, TDataType)

	template<typename TDataType>
	FixedPoints<TDataType>::FixedPoints()
		: ConstraintModule()
	{
	}

	template<typename TDataType>
	FixedPoints<TDataType>::~FixedPoints()
	{

	}


	template<typename TDataType>
	void FixedPoints<TDataType>::addPoint(int id)
	{
		m_ids.push_back(id);
	}

	template <typename Coord>
	__global__ void FixPoints(
		DeviceArray<Coord> curPos,
		DeviceArray<Coord> curVel,
		DeviceArray<Coord> iniPos, 
		DeviceArray<int> ids)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= ids.size()) return;
		
		int pnum = curPos.size();
		int fixId = ids[pId];
		if (fixId >= pnum)
		{
			return;
		}
		
		curPos[fixId] = iniPos[fixId];
		curVel[fixId] = Coord(0);
	}

	template<typename TDataType>
	void FixedPoints<TDataType>::constrain()
	{
		if (m_ids.size() <= 0)
			return;

		if (m_device_ids.size() != m_ids.size())
		{
			m_device_ids.resize(m_ids.size());
			Function1Pt::Copy(m_device_ids, m_ids);
		}

		auto mstate = getParent()->getMechanicalState();
		if (mstate->getMaterialType() == MechanicalState::RIGIDBODY)
		{
			auto init_poss = mstate->getField<HostVariable<Coord>>(MechanicalState::init_position())->getValue();
			mstate->getField<HostVariable<Coord>>(MechanicalState::position())->setValue(init_poss);
			mstate->getField<HostVariable<Coord>>(MechanicalState::velocity())->setValue(Coord(0));
		}
		else
		{
			auto init_poss = mstate->getField<DeviceBuffer<Coord>>(MechanicalState::init_position())->getValue();
			auto poss = mstate->getField<DeviceBuffer<Coord>>(MechanicalState::position())->getValue();
			auto vels = mstate->getField<DeviceBuffer<Coord>>(MechanicalState::velocity())->getValue();

			uint pDims = cudaGridSize(m_device_ids.size(), BLOCK_SIZE);

			FixPoints<Coord> << < pDims, BLOCK_SIZE >> > (poss, vels, init_poss, m_device_ids);
		}
	}

}