#include "Physika_Core/Utilities/CudaRand.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "BoundaryManager.h"
#include "Physika_Framework/Framework/Node.h"

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
		if (pId >= attArr.size()) return;

		status[pId] = NeedConstrain(attArr[pId]) ? 1 : 0;
	}


	template<typename TDataType>
	BoundaryManager<TDataType>::BoundaryManager()
		: ConstraintModule()
		, m_bConstrained(NULL)
	{
		initArgument(&m_position, "Position", "CUDA array used to store particles' positions");
		initArgument(&m_velocity, "Velocity", "CUDA array used to store particles' velocities");
		initArgument(&m_attribute, "Attribute", "CUDA array used to store particles' attributes");
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
		DeviceArray<Coord>* posArr = m_position.getField().getDataPtr();
		DeviceArray<Coord>* velArr = m_velocity.getField().getDataPtr();
		DeviceArray<Attribute>* attArr = m_attribute.getField().getDataPtr();

		Real dt = getParent()->getDt();

		Constrain(*posArr, *velArr, *attArr, dt);
		return true;
	}


	template<typename TDataType>
	void BoundaryManager<TDataType>::Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<Attribute>& attArr, float dt)
	{
		int pNum = posArr.size();
		if (m_bConstrained == NULL)
		{
			m_bConstrained = DeviceBuffer<int>::create(pNum);
		}
		DeviceArray<int>* stat = m_bConstrained->getDataPtr();
		cuint pDims = cudaGridSize(m_bConstrained->size(), BLOCK_SIZE);
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

		if (pId >= posArr.size()) return;
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
		cuint pDim = cudaGridSize(posArr.size(), BLOCK_SIZE);
		K_Constrain << <pDim, BLOCK_SIZE >> > (posArr, velArr, attArr, *distancefield3d, normalFriction, tangentialFriction, dt);
	}
}