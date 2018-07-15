#include "Physika_Core/Utilities/CudaRand.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "BoundaryManager.h"

namespace Physika {

	__device__ bool NeedConstrain(Attribute& att)
	{
		if (att.IsDynamic())
			return true;

		return false;
	}

	__global__ void BM_CheckStatues(
		DeviceArray<int> status, 
		DeviceArray<Attribute> attArr)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attArr.Size()) return;

		status[pId] = NeedConstrain(attArr[pId]) ? 1 : 0;
	}

	template<typename TDataType>
	BoundaryManager<TDataType>::BoundaryManager(ParticleSystem<TDataType>* parent)
		:Module()
		, m_parent(parent)
	{
		int pNum = m_parent->GetParticleNumber();
		m_bConstrained = DeviceBuffer<int>::create(pNum);
	}

	template<typename TDataType>
	BoundaryManager<TDataType>::~BoundaryManager(void)
	{
		const int nbarriers = size();
		for (int i = 0; i < nbarriers; i++) {
			delete m_barriers[i];
		}
		m_barriers.clear();
	}

	template<typename TDataType>
	bool BoundaryManager<TDataType>::execute()
	{
		Array<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		Array<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		Array<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();

		float dt = m_parent->getDt();

		Constrain(*posArr, *velArr, *attArr, dt);
		return true;
	}

	template<typename TDataType>
	void BoundaryManager<TDataType>::Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<Attribute>& attArr, float dt)
	{
		DeviceArray<int>* stat = m_bConstrained->getDataPtr();
		uint pDims = cudaGridSize(m_bConstrained->size(), BLOCK_SIZE);
		BM_CheckStatues << <pDims, BLOCK_SIZE >> > (*stat, attArr);

		for (int i = 0; i < m_barriers.size(); i++)
		{
			m_barriers[i]->Constrain(posArr, velArr, *stat, dt);
		}
	}

	// void Barrier::Inside(const float3& in_pos) const
	// {
	// 	float3 pos = in_pos;
	// 	pos -= Config::rotation_center;
	// // 	pos = float3(cos(angle)*pos.x - sin(angle)*pos.y, sin(angle)*pos.x + cos(angle)*pos.y, pos.z);
	// // 	pos += Config::rotation_center;

	// 	float dist;
	// 	float3 normal;
	// 	GetDistanceAndNormal(pos,dist,normal);
	// 	return (dist > 0);
	// }

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_Constrain(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> velArr,
		DeviceArray<int> attArr, 
		DistanceField3D<TDataType> df,
		Real normalFriction, 
		Real tangentialFriction,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= posArr.Size()) return;
		if (attArr[pId] == 0) return;

		Coord pos = posArr[pId];
		Coord vec = velArr[pId];

		Real dist;
		Coord normal;
		df.GetDistance(pos, dist, normal);
		// constrain particle
		if (dist <= 0) {
			Real olddist = -dist;
			Physika::RandNumber rGen(pId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			pos -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = vec.norm();
			Real vec_n = vec.dot(normal);
			Coord vec_normal = vec_n*normal;
			Coord vec_tan = vec - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			vec = vec_normal + vec_tan;
			vec *= pow(Real(M_E), -dt*tangentialFriction);
		}

		posArr[pId] = pos;
		velArr[pId] = vec;
	}

	template<typename TDataType>
	void BarrierDistanceField3D<TDataType>::Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<int>& attArr, Real dt)
	{
		uint pDim = cudaGridSize(posArr.Size(), BLOCK_SIZE);
		K_Constrain << <pDim, BLOCK_SIZE >> > (posArr, velArr, attArr, *distancefield3d, normalFriction, tangentialFriction, dt);
	}
}