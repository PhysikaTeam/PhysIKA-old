#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
#include "FixedPoints.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(FixedPoints, TDataType)

	template<typename TDataType>
	FixedPoints<TDataType>::FixedPoints()
		: ConstraintModule()
		, m_initPosID(MechanicalState::init_position())
	{
	}

	template<typename TDataType>
	FixedPoints<TDataType>::~FixedPoints()
	{
		m_ids.clear();
		m_device_ids.release();
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::addPoint(int id)
	{
		m_ids.push_back(id);
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::clear()
	{
		m_ids.clear();
	}

	template <typename Coord>
	__global__ void K_DoFixPoints(
		DeviceArray<Coord> curPos,
		DeviceArray<Coord> curVel,
		DeviceArray<Coord> iniPos,
		DeviceArray<int> ids)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= ids.size()) return;
		
		int num = curPos.size();
		int fixId = ids[pId];
		if (fixId >= num)
		{
			return;
		}
		
		curPos[fixId] = iniPos[fixId];
		curVel[fixId] = Coord(0);
	}

	template<typename TDataType>
	bool FixedPoints<TDataType>::constrain()
	{
		if (m_ids.size() <= 0)
			return false;

		if (m_device_ids.size() != m_ids.size())
		{
			m_device_ids.resize(m_ids.size());
			Function1Pt::copy(m_device_ids, m_ids);
		}

		auto mstate = getParent()->getMechanicalState();
		if (mstate->getMaterialType() == MechanicalState::RIGIDBODY)
		{
			auto init_poss = mstate->getField<HostVarField<Coord>>(m_initPosID)->getValue();
			mstate->getField<HostVarField<Coord>>(m_posID)->setValue(init_poss);
			mstate->getField<HostVarField<Coord>>(m_velID)->setValue(Coord(0));
		}
		else
		{
			auto init_poss = mstate->getField<DeviceArrayField<Coord>>(m_initPosID)->getValue();
			auto poss = mstate->getField<DeviceArrayField<Coord>>(m_posID)->getValue();
			auto vels = mstate->getField<DeviceArrayField<Coord>>(m_velID)->getValue();

			uint pDims = cudaGridSize(m_device_ids.size(), BLOCK_SIZE);

			K_DoFixPoints<Coord> << < pDims, BLOCK_SIZE >> > (poss, vels, init_poss, m_device_ids);
		}

		return true;
	}

}