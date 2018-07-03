#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/template_functions.h"
#include "ParticlePrediction.h"

namespace Physika
{
// 	struct PP_STATE
// 	{
// 		float3 bodyForce;
// 	};
// 
// 	__constant__ PP_STATE const_pp_state;

	template<typename Coord>
	__global__ void PP_Predict(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> velArr,
		Coord bodyForce,
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		Coord pos_i = posArr[pId];
		Coord vel_i = velArr[pId];

		pos_i += vel_i*dt;
		vel_i += bodyForce*dt;
//		vel_i += 20.0f*(make_float3(0.5f, 0.2f, 0.5f) - posArr[pId])*dt;

		posArr[pId] = pos_i;
		velArr[pId] = vel_i;
	}

	template<typename Coord>
	__global__ void PP_PredictPosition(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> velArr,
		DeviceArray<Attribute> attriArr, 
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		if (!attriArr[pId].IsFixed())
		{
			Coord pos_i = posArr[pId];
			Coord vel_i = velArr[pId];

			pos_i += vel_i*dt;

//			vel_i += 20.0f*(make_float3(0.5f) - posArr[pId])*dt;

#ifdef SIMULATION2D
			pos_i.z = 0.5f;
#endif
			posArr[pId] = pos_i;
			velArr[pId] = vel_i;

// 			if (attriArr[pId].IsPassive() && pos_i.y < 0.85f)
// 			{
// 				attriArr[pId].SetDynamic();
// 			}
		}
	}

	template<typename Coord>
	__global__ void PP_PredictVelocity(
		DeviceArray<Coord> velArr,
		DeviceArray<Attribute> attriArr, 
		Coord bodyForce,
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.Size()) return;

		if (attriArr[pId].IsDynamic())
		{
			Coord vel_i = velArr[pId];
			vel_i += bodyForce*dt;

			velArr[pId] = vel_i;
		}
	}

	template<typename Coord>
	__global__ void PP_CorrectPosition(
		DeviceArray<Coord> newPos, 
		DeviceArray<Coord> oldPos,
		DeviceArray<Coord> newVel,
		DeviceArray<Coord> oldVel,
		DeviceArray<Attribute> attriArr, 
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= newPos.Size()) return;

		if (!attriArr[pId].IsFixed())
		{
			newPos[pId] = oldPos[pId] + 0.5f * dt * (oldVel[pId] + newVel[pId]);

#ifdef SIMULATION2D
			newPos[pId].z = 0.5f;
#endif
		}
	}

	template<typename TDataType>
	ParticlePrediction<TDataType>::ParticlePrediction(ParticleSystem<TDataType>* parent)
		:Module()
		,m_parent(parent)
	{
		assert(m_parent != NULL);

		setInputSize(1);
		setOutputSize(1);

		updateStates();
	}

	template<typename TDataType>
	bool ParticlePrediction<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attriArr = m_parent->GetAttributeBuffer()->getDataPtr();
		Coord gravity = m_parent->GetGravity();
		float dt = m_parent->getDt();
		
		uint pDims = cudaGridSize(posArr->Size(), BLOCK_SIZE);
		
		PP_Predict <Coord> << <pDims, BLOCK_SIZE >> > (*posArr, *velArr, gravity, dt);

		return true;
	}

	template<typename TDataType>
	void ParticlePrediction<TDataType>::PredictPosition(float dt)
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attriArr = m_parent->GetAttributeBuffer()->getDataPtr();

		uint pDims = cudaGridSize(posArr->Size(), BLOCK_SIZE);
		PP_PredictPosition <Coord> << <pDims, BLOCK_SIZE >> > (*posArr, *velArr, *attriArr, dt);
	}

	template<typename TDataType>
	void ParticlePrediction<TDataType>::PredictVelocity(float dt)
	{
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attriArr = m_parent->GetAttributeBuffer()->getDataPtr();

		uint pDims = cudaGridSize(velArr->Size(), BLOCK_SIZE);
		Coord gravity = Make<Coord>(0);
		PP_PredictVelocity <Coord> << <pDims, BLOCK_SIZE >> > (*velArr, *attriArr, gravity, dt);
	}

	template<typename TDataType>
	void ParticlePrediction<TDataType>::CorrectPosition(float dt)
	{
		DeviceArray<Coord>* oldPos = m_parent->GetOldPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* newPos = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* oldVel = m_parent->GetOldVelocityBuffer()->getDataPtr();
		DeviceArray<Coord>* newVel = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attriArr = m_parent->GetAttributeBuffer()->getDataPtr();

		uint pDims = cudaGridSize(oldPos->Size(), BLOCK_SIZE);
		PP_CorrectPosition << <pDims, BLOCK_SIZE >> > (*newPos, *oldPos, *newVel, *oldVel, *attriArr, dt);
	}

	template<typename TDataType>
	bool ParticlePrediction<TDataType>::updateStates()
	{
// 		PP_STATE cm;
// 		cm.bodyForce = m_parent->GetBodyForce();
// 
// 		cudaMemcpyToSymbol(const_pp_state, &cm, sizeof(PP_STATE));

		return true;
	}

}