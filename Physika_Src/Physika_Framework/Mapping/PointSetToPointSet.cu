#pragma once
#include "PointSetToPointSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Topology/NeighborQuery.h"

namespace Physika
{
	template<typename TDataType>
	PointSetToPointSet<TDataType>::PointSetToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	PointSetToPointSet<TDataType>::PointSetToPointSet(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_from = from;
		m_to = to;
	}

	template<typename TDataType>
	PointSetToPointSet<TDataType>::~PointSetToPointSet()
	{

	}


	template<typename TDataType>
	bool PointSetToPointSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

	template <typename Real>
	__device__ Real D_Weight(Real r, Real h)
	{
		Real q = r / h;
		if (r > h)
		{
			return Real(0);
		}
		return 1 - q*q;
	}


	template <typename Real, typename Coord>
	__global__ void K_ApplyTransform(
		DeviceArray<Coord> to,
		DeviceArray<Coord> from,
		DeviceArray<Coord> initTo,
		DeviceArray<Coord> initFrom,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= to.size()) return;

		Real totalWeight = 0;
		Coord to_i = to[pId];
		Coord initTo_i = initTo[pId];
		Coord accDisplacement_i = Coord(0);
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (initTo_i - initFrom[j]).norm();
			Real weight = D_Weight(r, smoothingLength);

			//printf("%f, %f, %f : %f %f %f  \n", initTo_i[0], initTo_i[1], initTo_i[2], initFrom[j][0], initFrom[j][1], initFrom[j][2]);

			totalWeight += weight;
			accDisplacement_i += (from[j] - initFrom[j])*weight;
		}

		accDisplacement_i = totalWeight > EPSILON ? (accDisplacement_i / totalWeight) : accDisplacement_i;
		to[pId] = initTo_i + accDisplacement_i;

		//printf("%f, %f, %f \n", accDisplacement_i[0], accDisplacement_i[1], accDisplacement_i[2]);
	}

	template<typename TDataType>
	bool PointSetToPointSet<TDataType>::apply()
	{
		cuint pDim = cudaGridSize(m_to->getPoints().size(), BLOCK_SIZE);

		K_ApplyTransform << <pDim, BLOCK_SIZE >> > (
			m_to->getPoints(),
			m_from->getPoints(),
			m_initTo->getPoints(),
			m_initFrom->getPoints(),
			m_neighborhood,
			m_radius);

		return true;
	}

	template<typename TDataType>
	void PointSetToPointSet<TDataType>::match(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<PointSet<TDataType>>();
		m_initTo = std::make_shared<PointSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);

		NeighborQuery<TDataType>* nbQuery = new NeighborQuery<TDataType>(m_initFrom->getPoints());

		m_neighborhood.resize(m_initTo->getPoints().size());
		nbQuery->queryParticleNeighbors(m_neighborhood, m_initTo->getPoints(), m_radius);

		delete nbQuery;
	}
}