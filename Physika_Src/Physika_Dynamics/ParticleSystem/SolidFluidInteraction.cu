#include "SolidFluidInteraction.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "ParticleSystem.h"
#include "Physika_Framework/Topology/NeighborQuery.h"
#include "Kernel.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(SolidFluidInteraction, TDataType)


	template<typename TDataType>
	Physika::SolidFluidInteraction<TDataType>::SolidFluidInteraction(std::string name)
		:Node(name)
	{
		
	}

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::~SolidFluidInteraction()
	{
		
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::initialize()
	{
		int total_num = 0;
		std::vector<int> ids;
		std::vector<Real> mass;
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			auto points = m_particleSystems[i]->getPosition()->getValue();
			total_num += points.size();
			Real m = m_particleSystems[i]->getMass() / points.size();
			for (int j = 0; j < points.size(); j++)
			{
				ids.push_back(i);
				mass.push_back(m);
			}
		}

		m_objId.resize(total_num);
		m_position.setElementCount(total_num);
		m_vels.resize(total_num);
		m_mass.resize(total_num);

		posBuf.resize(total_num);
		weights.resize(total_num);
		init_pos.resize(total_num);

		Function1Pt::copy(m_objId, ids);
		Function1Pt::copy(m_mass, mass);
		ids.clear();
		mass.clear();

		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_position.connect(m_nbrQuery->m_position);
		m_nbrQuery->initialize();
		
		return true;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		return nullptr;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		this->addChild(child);
		m_particleSystems.push_back(child);

		return nullptr;
	}

	template<typename Real, typename Coord>
	__global__ void K_Collide(
		DeviceArray<int> objIds,
		DeviceArray<Real> mass,
		DeviceArray<Coord> points,
		DeviceArray<Coord> newPoints,
		DeviceArray<Real> weights,
		NeighborList<int> neighbors,
		Real radius
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		SpikyKernel<Real> kernel;

		Real r;
		Coord pos_i = points[pId];
		int id_i = objIds[pId];
		Real mass_i = mass[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		int col_num = 0;
		Coord pos_num = Coord(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord pos_j = points[j];

			r = (pos_i - pos_j).norm();
			if (r < radius && objIds[j] != id_i)
			{
				col_num++;
				Real mass_j = mass[j];
				Coord center = (pos_i + pos_j) / 2;
				Coord n = pos_i - pos_j;
				n = n.norm() < EPSILON ? Coord(0, 0, 0) : n.normalize();

				Real a = mass_i / (mass_i + mass_j);

				Real d = radius - r;

				Coord target_i = pos_i + (1 - a)*d*n;// (center + 0.5*radius*n);
				Coord target_j = pos_j - a*d*n;// (center - 0.5*radius*n);
				//				pos_num += (center + 0.4*radius*n);

				Real weight = kernel.Weight(r, 2 * radius);

				atomicAdd(&newPoints[pId][0], weight*target_i[0]);
				atomicAdd(&newPoints[j][0], weight*target_j[0]);

				atomicAdd(&weights[pId], weight);
				atomicAdd(&weights[j], weight);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&newPoints[pId][1], weight*target_i[1]);
					atomicAdd(&newPoints[j][1], weight*target_j[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&newPoints[pId][2], weight*target_i[2]);
					atomicAdd(&newPoints[j][2], weight*target_j[2]);
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

		velocites[pId] += 0.5*(curPoints[pId] - initPoints[pId]) / dt;
	}

	template<typename TDataType>
	void SolidFluidInteraction<TDataType>::advance(Real dt)
	{
		int start = 0;
		DeviceArray<Coord>& allpoints = m_position.getValue();
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			DeviceArray<Coord>& points = m_particleSystems[i]->getPosition()->getValue();
			DeviceArray<Coord>& vels = m_particleSystems[i]->getVelocity()->getValue();
			int num = points.size();
			cudaMemcpy(allpoints.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_vels.getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start += num;
		}

		m_nbrQuery->compute();

		Function1Pt::copy(init_pos, allpoints);

		Real radius = 0.005;

		uint pDims = cudaGridSize(allpoints.size(), BLOCK_SIZE);
		for (size_t it = 0; it < 5; it++)
		{
			weights.reset();
			posBuf.reset();
			K_Collide << <pDims, BLOCK_SIZE >> > (
				m_objId, 
				m_mass,
				allpoints,
				posBuf, 
				weights, 
				m_nbrQuery->getNeighborList(),
				radius);

			K_ComputeTarget << <pDims, BLOCK_SIZE >> > (
				allpoints,
				posBuf, 
				weights);

			Function1Pt::copy(allpoints, posBuf);
		}

		K_ComputeVelocity << <pDims, BLOCK_SIZE >> > (init_pos, allpoints, m_vels, getParent()->getDt());

		start = 0;
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			DeviceArray<Coord>& points = m_particleSystems[i]->getPosition()->getValue();
			DeviceArray<Coord>& vels = m_particleSystems[i]->getVelocity()->getValue();
			int num = points.size();
			cudaMemcpy(points.getDataPtr(), allpoints.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(vels.getDataPtr(), m_vels.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

			start += num;
		}

	}
}