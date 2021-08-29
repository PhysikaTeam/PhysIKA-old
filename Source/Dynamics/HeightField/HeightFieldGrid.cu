#include "HeightFieldGrid.h"
//#include <fstream>
//#include <iostream>
//#include <sstream>
//#include "Core/Utility.h"
//
//namespace PhysIKA
//{
//
//
//	template<typename TDataType>
//	HeightFieldGrid<TDataType>::HeightFieldGrid()
//	{
//	}
//
//	template<typename TDataType>
//	HeightFieldGrid<TDataType>::~HeightFieldGrid()
//	{
//	}
//
//	template<typename TDataType>
//	void PhysIKA::HeightFieldGrid<TDataType>::setSpace(Real dx, Real dz)
//	{
//		m_dx = dx;
//		m_dz = dz;
//	}
//
//	template<typename TDataType>
//	void HeightFieldGrid<TDataType>::copyFrom(HeightFieldGrid<TDataType>& pointSet)
//	{
//	}
//
//
//	template <typename Real, typename Coord>
//	__global__ void PS_Scale(
//		DeviceArray<Coord> vertex,
//		Real s)
//	{
//		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
//		if (pId >= vertex.size()) return;
//		//return;
//		vertex[pId] = vertex[pId] * s;
//	}
//
//	template<typename TDataType>
//	void HeightFieldGrid<TDataType>::scale(Real s)
//	{
//		//cuExecute(m_coords.size(), PS_Scale, m_coords, s);
//	}
//
//	template <typename Coord>
//	__global__ void PS_Scale(
//		DeviceArray<Coord> vertex,
//		Coord s)
//	{
//		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
//		if (pId >= vertex.size()) return;
//
//		Coord pos_i = vertex[pId];
//		vertex[pId] = Coord(pos_i[0] * s[0], pos_i[1] * s[1], pos_i[2] * s[2]);
//	}
//
//	template<typename TDataType>
//	void PhysIKA::HeightFieldGrid<TDataType>::scale(Coord s)
//	{
//		//cuExecute(m_coords.size(), PS_Scale, m_coords, s);
//	}
//
//
//
//
//	template<typename TDataType>
//	void PhysIKA::HeightFieldGrid<TDataType>::translate(Coord t)
//	{
//		//cuExecute(m_coords.size(), PS_Translate, m_coords, t);
//
//// 		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);
////
//// 		PS_Translate << <pDims, BLOCK_SIZE >> > (
//// 			m_coords,
//// 			t);
//// 		cuSynchronize();
//	}
//}