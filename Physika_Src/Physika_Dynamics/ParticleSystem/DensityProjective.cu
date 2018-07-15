#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "DensityProjective.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Core/Vectors/vector.h"
//#include "Physika_Core/Utilities/template_functions.h"

namespace Physika
{
// 	struct DP_STATE
// 	{
// 		float mass;
// 		float smoothingLength;
// 		float weight;
// 		SpikyKernel kernSpiky;
// 	};
// 
// 	__constant__ DP_STATE const_dp_state;

	template<typename Real, typename Coord>
	__global__ void DP_ComputeLambdas(
		DeviceArray<Real> lambdaArr, 
		DeviceArray<Real> rhoArr, 
		DeviceArray<Coord> posArr, 
		DeviceArray<NeighborList> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = 0;
		Coord grad_ci(0);

		SpikyKernel<Real> kern;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template<typename Real, typename Coord>
	__global__ void DP_ComputeDisplacement(
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas,
		DeviceArray<Coord> posArr, 
		DeviceArray<NeighborList> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		SpikyKernel<Real> kern;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 5.0f*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], -dp_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], -dp_ij[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], -dp_ij[2]);
				}
			}
		}
	}

	template<typename Real, typename Coord>
	__global__ void DP_UpdatePosition(
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> tmpPos, 
		DeviceArray<Coord> dPos, 
		DeviceArray<Attribute> attArr,
		Real mass,
		Real w,
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		if (attArr[pId].IsDynamic())
		{
//			Real w = const_dp_state.weight;
			Coord newPos = posArr[pId] + dPos[pId];
			Real a = mass/dt/dt + w;
			a = w / a;
			Real b = 1.0f - a;

			posArr[pId] = a*newPos + b*tmpPos[pId];
		}
	}

	template<typename Real, typename Coord>
	__global__ void DP_UpdateVelocity(
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> tmpArr, 
		DeviceArray<Attribute> attArr, 
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		if (attArr[pId].IsDynamic())
		{
			velArr[pId] += (posArr[pId] - tmpArr[pId]) / dt;
		}
	}

	template<typename TDataType>
	DensityProjective<TDataType>::DensityProjective(ParticleSystem<TDataType>* parent)
		:DensityConstraint(parent)
		,m_parent(parent)
		,m_maxIteration(1)
		,m_w(10.0f)
	{
		assert(m_parent != NULL);

		setInputSize(2);
		setOutputSize(1);

		int num = m_parent->GetParticleNumber();

		m_lamda = DeviceBuffer<Real>::create(num);
		m_deltaPos = DeviceBuffer<Coord>::create(num);
		m_posTmp = DeviceBuffer<Coord>::create(num);

		updateStates();
	}

	template<typename TDataType>
	bool DensityProjective<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Real>* rhoArr = m_parent->GetDensityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();
		float dt = m_parent->getDt();

		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();

		DeviceArray<Real>* lamda = m_lamda->getDataPtr();
		DeviceArray<Coord>* deltaPos = m_deltaPos->getDataPtr();
		DeviceArray<Coord>* tmpPos = m_posTmp->getDataPtr();

		Real mass = m_parent->GetParticleMass();
		Real smoothingLength = m_parent->GetSmoothingLength();

		dim3 pDims = int(ceil(posArr->Size() / BLOCK_SIZE + 0.5f));

		Module* densitySum = m_parent->getModule("SummationDensity");

		Function1Pt::Copy(*tmpPos, *posArr);

		int it = 0;
		while (it < 1)
		{
			deltaPos->Reset();

			densitySum->execute();
			DP_ComputeLambdas <Real, Coord> << <pDims, BLOCK_SIZE >> > (*lamda, *rhoArr, *posArr, *neighborArr,smoothingLength);
			DP_ComputeDisplacement <Real, Coord> << <pDims, BLOCK_SIZE >> > (*deltaPos, *lamda, *posArr, *neighborArr, smoothingLength);
			DP_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (*posArr, *tmpPos, *deltaPos, *attArr, mass, m_w, dt);

			it++;
		}

		DP_UpdateVelocity <Real, Coord> << <pDims, BLOCK_SIZE >> > (*velArr, *posArr, *tmpPos, *attArr, dt);

		return true;
	}

	template<typename TDataType>
	bool DensityProjective<TDataType>::updateStates()
	{
// 		DP_STATE cm;
// 		cm.mass = m_parent->GetParticleMass();
// 		cm.smoothingLength = m_parent->GetSmoothingLength();
// 		cm.weight = m_w;
// 		cm.kernSpiky = SpikyKernel();
// 		cudaMemcpyToSymbol(const_dp_state, &cm, sizeof(DP_STATE));

		return true;
	}

}