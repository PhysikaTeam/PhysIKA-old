#include "RodCollision.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/CollidableObject.h"
#include "Framework/Collision/CollidablePoints.h"
#include "Framework/Topology/NeighborQuery.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(RodCollision, TDataType)

	template<typename TDataType>
	RodCollision<TDataType>::RodCollision()
		: CollisionModel()
	{
	}

	template<typename TDataType>
	RodCollision<TDataType>::~RodCollision()
	{
		m_collidableObjects.clear();
	}

	template<typename TDataType>
	bool RodCollision<TDataType>::isSupport(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			return true;
		}
		return false;
	}


	template<typename TDataType>
	void RodCollision<TDataType>::addCollidableObject(std::shared_ptr<CollidableObject> obj)
	{
		auto derived = std::dynamic_pointer_cast<CollidablePoints<TDataType>>(obj);
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			m_collidableObjects.push_back(derived);
		}
	}

	template<typename Real, typename Coord>
	__global__ void K_Collide(
		DeviceArray<int> objIds,
		DeviceArray<Coord> points,
		DeviceArray<Coord> newPoints,
		DeviceArray<Real> weights,
		NeighborList<int> neighbors,
		Real radius
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		Real r;
		Coord pos_i = points[pId];
		int id_i = objIds[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		int col_num = 0;
		Coord pos_num = Coord(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - points[j]).norm();
			if (r < radius && objIds[j] != id_i)
			{
				col_num++;
				Coord center = (pos_i + points[j]) / 2;
				Coord n = pos_i - center;
				if (n.norm() < EPSILON)
					n = Coord(1, 0, 0);
				else
				{
					n = n.normalize();
				}

				Coord target_i = (center + 0.5*radius*n);
				Coord target_j = (center - 0.5*radius*n);
//				pos_num += (center + 0.4*radius*n);

				atomicAdd(&newPoints[pId][0], target_i[0]);
				atomicAdd(&newPoints[j][0], target_j[0]);

				atomicAdd(&weights[pId], Real(1));
				atomicAdd(&weights[j], Real(1));

				if (Coord::dims() >= 2)
				{
					atomicAdd(&newPoints[pId][1], target_i[1]);
					atomicAdd(&newPoints[j][1], target_j[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&newPoints[pId][2], target_i[2]);
					atomicAdd(&newPoints[j][2], target_j[2]);
				}
			}
		}

//		if (col_num != 0)
//			pos_num /= col_num;
//		else
//			pos_num = pos_i;
//
//		newPoints[pId] = pos_num;
	}

	template<typename Real, typename Coord>
	__global__ void K_ComputeTarget(
		DeviceArray<Coord> oldPoints,
		DeviceArray<Coord> newPoints, 
		DeviceArray<Real> weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= oldPoints.size()) return;

		if (weights[pId] > EPSILON)
		{
			newPoints[pId] /= weights[pId];
		}
		else
			newPoints[pId] = oldPoints[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_ComputeVelocity(
		DeviceArray<Coord> initPoints,
		DeviceArray<Coord> curPoints,
		DeviceArray<Coord> velocites,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocites.size()) return;

		velocites[pId] += (curPoints[pId] - initPoints[pId]) / dt;
	}

	template<typename TDataType>
	void RodCollision<TDataType>::doCollision()
	{
		int start = 0;
		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			DeviceArray<Coord>& points = m_collidableObjects[i]->getPositions();
			DeviceArray<Coord>& vels = m_collidableObjects[i]->getVelocities();
			int num = points.size();
			cudaMemcpy(m_points.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_vels.getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start += num;
		}

		if (m_nbrQuery == nullptr)
		{
			m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		}
		if (m_nList == nullptr)
		{
			m_nList = std::make_shared<NeighborList<int>>();
			m_nList->resize(m_points.size());
			m_nList->setNeighborLimit(5);
		}
		

		Real radius = 0.005;
		m_nbrQuery->queryParticleNeighbors(*m_nList, m_points, radius);

		DeviceArray<Coord> posBuf;
		posBuf.resize(m_points.size());

		DeviceArray<Real> weights;
		weights.resize(m_points.size());

		DeviceArray<Coord> init_pos;
		init_pos.resize(m_points.size());

		Function1Pt::copy(init_pos, m_points);

		uint pDims = cudaGridSize(m_points.size(), BLOCK_SIZE);
		for (size_t it = 0; it < 5; it++)
		{
			weights.reset();
			posBuf.reset();
			K_Collide << <pDims, BLOCK_SIZE >> > (m_objId, m_points, posBuf, weights, *m_nList, radius);
			K_ComputeTarget << <pDims, BLOCK_SIZE >> > (m_points, posBuf, weights);
			Function1Pt::copy(m_points, posBuf);
		}

		K_ComputeVelocity << <pDims, BLOCK_SIZE >> > (init_pos, m_points, m_vels, getParent()->getDt());

		posBuf.release();
		weights.release();
		init_pos.release();

		start = 0;
		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			DeviceArray<Coord>& points = m_collidableObjects[i]->getPositions();
			DeviceArray<Coord>& vels = m_collidableObjects[i]->getVelocities();
			int num = points.size();
			cudaMemcpy(points.getDataPtr(), m_points.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(vels.getDataPtr(), m_vels.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

			m_collidableObjects[i]->updateMechanicalState();
			start += num;
		}
	}


	template<typename TDataType>
	bool RodCollision<TDataType>::initializeImpl()
	{
		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			m_collidableObjects[i]->initialize();
		}

		int totalNum = 0;
		std::vector<int> ids;
		std::vector<Coord> hPos;
		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			DeviceArray<Coord>& points = m_collidableObjects[i]->getPositions();
			for (int j = 0; j < points.size(); j++)
			{
				ids.push_back(i);
			}
			totalNum += points.size();
		}

		if (totalNum <= 0)
			return false;

		m_objId.resize(totalNum);
		m_points.resize(totalNum);
		m_vels.resize(totalNum);

		Function1Pt::copy(m_objId, ids);

		return true;
	}

}