#include "PointSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(PointSet, TDataType)

	template<typename TDataType>
	PointSet<TDataType>::PointSet()
		: TopologyModule()
		, m_samplingDistance(Real(0.1))
	{
		std::vector<Coord> positions;
		for (Real x = -2.0; x < 2.0; x += m_samplingDistance) {
			for (Real y = -2.0; y < 2.0; y += m_samplingDistance) {
				for (Real z = -2.0; z < 2.0; z += m_samplingDistance) {
					positions.push_back(Coord(Real(x), Real(y), Real(z)));
				}
			}
		}
		this->setPoints(positions);

		m_normals.resize(positions.size());
		m_normals.reset();
	}

	template<typename TDataType>
	PointSet<TDataType>::~PointSet()
	{
	}

	template<typename TDataType>
	bool PointSet<TDataType>::initializeImpl()
	{
		return true;
	}

	template<typename TDataType>
	void PointSet<TDataType>::copyFrom(PointSet<TDataType>& pointSet)
	{
		if (m_coords.size() != pointSet.getPointSize())
		{
			m_coords.resize(pointSet.getPointSize());
			m_normals.resize(pointSet.getPointSize());
		}
		Function1Pt::copy(m_coords, pointSet.getPoints());
		Function1Pt::copy(m_normals, pointSet.getNormals());
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(std::vector<Coord>& pos)
	{
		m_normals.resize(pos.size());
		m_coords.resize(pos.size());
		Function1Pt::copy(m_coords, pos);

		tagAsChanged();
	}

	template<typename TDataType>
	NeighborList<int>* PointSet<TDataType>::getPointNeighbors()
	{
		if (isTopologyChanged())
		{
			updatePointNeighbors();
		}

		return &m_pointNeighbors;
	}

	template<typename TDataType>
	void PointSet<TDataType>::updatePointNeighbors()
	{
		if (m_coords.isEmpty())
			return;
	}

	template <typename Real, typename Coord>
	__global__ void PS_Scale(
		DeviceArray<Coord> vertex,
		Real s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		vertex[pId] = vertex[pId] * s;
	}

	template<typename TDataType>
	void PointSet<TDataType>::scale(Real s)
	{
		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);

		PS_Scale<< <pDims, BLOCK_SIZE >> > (
			m_coords,
			s);
	}

	template <typename Coord>
	__global__ void PS_Translate(
		DeviceArray<Coord> vertex,
		Coord t)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		vertex[pId] = vertex[pId] + t;
	}


	template<typename TDataType>
	void Physika::PointSet<TDataType>::translate(Coord t)
	{
		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);

		PS_Translate << <pDims, BLOCK_SIZE >> > (
			m_coords,
			t);
	}
}