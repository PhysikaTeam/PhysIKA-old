#pragma once
#include "RigidToPoints.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

namespace Physika
{

	template<typename TDataType>
	RigidToPoints<TDataType>::RigidToPoints()
		: Mapping()
	{

	}


	template<typename TDataType>
	RigidToPoints<TDataType>::~RigidToPoints()
	{
		if (m_refPoints.getDataPtr() != NULL)
		{
			m_refPoints.release();
		}
	}


	template<typename TDataType>
	void RigidToPoints<TDataType>::initialize(Rigid& rigid, DeviceArray<Coord>& points)
	{
		m_refRigid = rigid;
		m_refPoints.resize(points.size());
		Function1Pt::copy(m_refPoints, points);
	}

	template <typename Coord, typename Rigid, typename Matrix>
	__global__ void ApplyRigidTranform(
		DeviceArray<Coord> points,
		Coord curCenter,
		Matrix curMat,
		DeviceArray<Coord> refPoints,
		Coord refCenter,
		Matrix refMat)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		points[pId] = curCenter + curMat*refMat.transpose()*(refPoints[pId] - refCenter);
	}

	template<typename TDataType>
	void RigidToPoints<TDataType>::applyTransform(Rigid& rigid, DeviceArray<Coord>& points)
	{
		if (points.size() != m_refPoints.size())
		{
			std::cout << "The array sizes does not match for RigidToPoints" << std::endl;
		}

		uint pDims = cudaGridSize(points.size(), BLOCK_SIZE);

		ApplyRigidTranform<Coord, Rigid, Matrix><< <pDims, BLOCK_SIZE >> >(points, rigid.getCenter(), rigid.getRotationMatrix(), m_refPoints, m_refRigid.getCenter(), m_refRigid.getRotationMatrix());
	}

	//TODO:
	template<typename TDataType>
	void RigidToPoints<TDataType>::applyInverseTransform(Rigid& rigid, DeviceArray<Coord>& points)
	{
		
	}


}