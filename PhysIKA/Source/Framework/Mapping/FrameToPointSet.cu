#pragma once
#include "FrameToPointSet.h"
#include "Core/Utility.h"
#include "Framework/Topology/Frame.h"
#include "Framework/Topology/PointSet.h"

namespace PhysIKA
{
	template<typename TDataType>
	FrameToPointSet<TDataType>::FrameToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	FrameToPointSet<TDataType>::FrameToPointSet(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
		: TopologyMapping()
	{
		m_from = from;
		m_to = to;
	}


	template<typename TDataType>
	FrameToPointSet<TDataType>::~FrameToPointSet()
	{
		if (m_refPoints.getDataPtr() != NULL)
		{
			m_refPoints.release();
		}
	}


	template<typename TDataType>
	void FrameToPointSet<TDataType>::initialize(const Rigid& rigid, DeviceArray<Coord>& points)
	{
		m_refRigid = rigid;
		m_refPoints.resize(points.size());
		Function1Pt::copy(m_refPoints, points);
	}


	template<typename TDataType>
	void FrameToPointSet<TDataType>::match(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<Frame<TDataType>>();
		m_initTo = std::make_shared<PointSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);
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
	void FrameToPointSet<TDataType>::applyTransform(const Rigid& rigid, DeviceArray<Coord>& points)
	{
		if (points.size() != m_refPoints.size())
		{
			std::cout << "The array sizes does not match for RigidToPoints" << std::endl;
		}

		uint pDims = cudaGridSize(points.size(), BLOCK_SIZE);

		ApplyRigidTranform<Coord, Rigid, Matrix><< <pDims, BLOCK_SIZE >> >(points, rigid.getCenter(), rigid.getRotationMatrix(), m_refPoints, m_refRigid.getCenter(), m_refRigid.getRotationMatrix());
	}

	template<typename TDataType>
	bool FrameToPointSet<TDataType>::apply()
	{
		DeviceArray<Coord>& m_coords = m_initTo->getPoints();

		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);

		ApplyRigidTranform<Coord, Rigid, Matrix> << <pDims, BLOCK_SIZE >> >(
			m_to->getPoints(),
			m_from->getCenter(), 
			m_from->getOrientation(),
			m_coords,
			m_initFrom->getCenter(), 
			m_initFrom->getOrientation());

		return true;
	}


	template<typename TDataType>
	bool FrameToPointSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

}