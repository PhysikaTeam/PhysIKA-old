#include <cuda_runtime.h>
#include "Physika_Core/Utilities/template_functions.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "DensityPBD.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(DensityPBD, TDataType)
// 	struct PBD_STATE
// 	{
// 		Real mass;
// 		Real smoothingLength;
// 		SpikyKernel kernSpiky;
// 	};

//	__constant__ PBD_STATE const_pbd_state;

	template <typename Real, typename Coord>
	__global__ void DC_ComputeLambdas(
		DeviceArray<Real> lambdaArr, 
		DeviceArray<Real> rhoArr, 
		DeviceArray<Coord> posArr, 
		DeviceArray<NeighborList> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		Coord pos_i = posArr[pId];

		SpikyKernel<Real> kern;

		Real lamda_i = Real(0);
		Coord grad_ci = Make<Coord>(Real(0));

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			Real r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += dot(g, g);
			}
		}

		lamda_i += dot(grad_ci, grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void DC_ComputeDisplacement(
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas, 
		DeviceArray<Coord> posArr, 
		DeviceArray<NeighborList> neighbors, 
		Real smoothingLength,
		float dt)
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
			Real r = length(pos_i - posArr[j]);
			if (r > EPSILON)
			{
				Coord dp_ij = 0.1*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
// 				atomicAdd(&dPos[pId].x, dp_ij.x);
// 				atomicAdd(&dPos[pId].y, dp_ij.y);
// 				atomicAdd(&dPos[pId].z, dp_ij.z);
				AtomicAdd(dPos[pId], dp_ij);
// 
// 				atomicAdd(&dPos[j].x, -dp_ij.x);
// 				atomicAdd(&dPos[j].y, -dp_ij.y);
// 				atomicAdd(&dPos[j].z, -dp_ij.z);
				AtomicAdd(dPos[j], -dp_ij);
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void DC_UpdatePosition(
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dPos, 
		DeviceArray<Attribute> attArr, 
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		if (attArr[pId].IsDynamic())
		{
			posArr[pId] += dPos[pId];
			velArr[pId] += dPos[pId] / dt;
		}
	}


	template<typename TDataType>
	Physika::DensityPBD<TDataType>::DensityPBD()
	{

	}

	template<typename TDataType>
	DensityPBD<TDataType>::DensityPBD(ParticleSystem<TDataType>* parent)
		:DensityConstraint(parent)
		,m_parent(parent)
		,m_maxIteration(1)
	{
		assert(m_parent != NULL);

		setInputSize(2);
		setOutputSize(1);

		int num = m_parent->GetParticleNumber();

		m_lamda = DeviceBuffer<Real>::create(num);
		m_deltaPos = DeviceBuffer<Coord>::create(num);

		updateStates();
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Real>* rhoArr = m_parent->GetDensityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();

		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();

		DeviceArray<Real>* lamda = m_lamda->getDataPtr();
		DeviceArray<Coord>* deltaPos = m_deltaPos->getDataPtr();

		float dt = m_parent->getDt();

		Real smoothingLength = m_parent->GetSmoothingLength();

		dim3 pDims = int(ceil(posArr->Size() / BLOCK_SIZE + 0.5f));

		Module* densitySum = m_parent->getModule("COMPUTE_DENSITY");

		int it = 0;
		while (it < 5)
		{
			deltaPos->Reset();

			densitySum->execute();
 			DC_ComputeLambdas <Real, Coord> << <pDims, BLOCK_SIZE >> > (*lamda, *rhoArr, *posArr, *neighborArr, smoothingLength);
 			DC_ComputeDisplacement <Real, Coord> << <pDims, BLOCK_SIZE >> > (*deltaPos, *lamda, *posArr, *neighborArr, smoothingLength, dt);
 			DC_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (*posArr, *velArr, *deltaPos, *attArr, dt);

			it++;
		}

		return true;
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::updateStates()
	{
// 		PBD_STATE cm;
// 		cm.mass = m_parent->GetParticleMass();
// 		cm.smoothingLength = m_parent->GetSmoothingLength();
// 		cm.kernSpiky = SpikyKernel();
// 		cudaMemcpyToSymbol(const_pbd_state, &cm, sizeof(PBD_STATE));

		return true;
	}

}