#include <cuda_runtime.h>
#include "Core/Utility.h"
#include "Framework/Framework/Log.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Framework/Node.h"
#include "FixedPoints.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(FixedPoints, TDataType)

	template<typename TDataType>
	FixedPoints<TDataType>::FixedPoints()
		: ConstraintModule()
	{
		this->attachField(&m_position, "position", "Storing the particle positions!", false);
		this->attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);

	}

	template<typename TDataType>
	FixedPoints<TDataType>::~FixedPoints()
	{
		m_bFixed.release();
		m_fixed_positions.release();
	}


	template<typename TDataType>
	bool FixedPoints<TDataType>::initializeImpl()
	{
		if (m_position.isEmpty() || m_velocity.isEmpty())
		{
			std::cout << "Exception: " << std::string("FixedPoints's fields are not fully initialized!") << "\n";
			return false;
		}

		return true;
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::updateContext()
	{
		int totalNum = m_position.getValue().size();
		if (m_bFixed.size() != totalNum)
		{
			m_bFixed_host.resize(totalNum);
			m_fixed_positions_host.resize(totalNum);

			m_bFixed.resize(totalNum);
			m_fixed_positions.resize(totalNum);
		}

		for (int i = 0; i < m_bFixed_host.size(); i++)
		{
			m_bFixed_host[i] = 0;
		}

		for (auto it = m_fixedPts.begin(); it != m_fixedPts.end(); it++)
		{
			if (it->first >= 0 && it->first < totalNum)
			{
				m_bFixed_host[it->first] = 1;
				m_fixed_positions_host[it->first] = it->second;
			}
		}

		Function1Pt::copy(m_bFixed, m_bFixed_host);
		Function1Pt::copy(m_fixed_positions, m_fixed_positions_host);
	}

	template<typename TDataType>
	void FixedPoints<TDataType>::addFixedPoint(int id, Coord pt)
	{
		m_fixedPts[id] = pt;

		bUpdateRequired = true;
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::removeFixedPoint(int id)
	{
		auto it = m_fixedPts.begin();
		while (it != m_fixedPts.end())
		{
			if (it->first == id)
			{
				m_fixedPts.erase(it++);
			}
			else
				it++;
		}

		bUpdateRequired = true;
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::clear()
	{
		m_fixedPts.clear();

		bUpdateRequired = true;
	}

	template <typename Coord>
	__global__ void K_DoFixPoints(
		DeviceArray<Coord> curPos,
		DeviceArray<Coord> curVel,
		DeviceArray<int> bFixed,
		DeviceArray<Coord> fixedPts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;
		
		if (bFixed[pId])
		{
			curPos[pId] = fixedPts[pId];
			curVel[pId] = Coord(0);
		}

	}

	template<typename TDataType>
	bool FixedPoints<TDataType>::constrain()
	{
		if (m_fixedPts.size() <= 0)
			return false;

		if (bUpdateRequired)
		{
			updateContext();
			bUpdateRequired = false;
		}


		uint pDims = cudaGridSize(m_bFixed.size(), BLOCK_SIZE);

		K_DoFixPoints<Coord> << < pDims, BLOCK_SIZE >> > (m_position.getValue(), m_velocity.getValue(), m_bFixed, m_fixed_positions);

		return true;
	}

	template <typename Coord>
	__global__ void K_DoPlaneConstrain(
		DeviceArray<Coord> curPos,
		Coord origin,
		Coord dir)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		float tmp = dir.dot(curPos[pId] - origin);
		if (tmp < 0)
		{
			curPos[pId] -= tmp*dir;
		}
	}

	template<typename TDataType>
	void PhysIKA::FixedPoints<TDataType>::constrainPositionToPlane(Coord pos, Coord dir)
	{
		uint pDims = cudaGridSize(m_bFixed.size(), BLOCK_SIZE);

		K_DoPlaneConstrain<< < pDims, BLOCK_SIZE >> > (m_position.getValue(), pos, dir);
	}

}