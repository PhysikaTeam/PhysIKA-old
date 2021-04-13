#include <cuda_runtime.h>
//#include "Core/Utilities/template_functions.h"
#include "Core/Utility.h"
#include "DensityPBD.h"
#include "Framework/Framework/Node.h"
#include <string>
#include "SummationDensity.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(DensityPBD, TDataType)

	template<typename Real,
			 typename Coord>
	__global__ void K_InitKernelFunction(
		DeviceArray<Real> weights,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kernel,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= weights.size()) return;

		Coord pos_i = posArr[pId];

		int nbSize = neighbors.getNeighborSize(pId);
		Real total_weight = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				total_weight += kernel.Weight(r, smoothingLength);
			}
		}

		weights[pId] = total_weight;
	}


	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdas(
		DeviceArray<Real> lambdaArr,
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

// 		if (pId < 20)
// 		{
// 			printf("%f \n", lamda_i);
// 		}

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdas(
		DeviceArray<Real> lambdaArr,
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		DeviceArray<Real> massInvArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g) * massInvArr[j];
			}
		}

		lamda_i += grad_ci.dot(grad_ci) * massInvArr[pId];

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}


	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacement(
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas, 
		DeviceArray<Coord> posArr, 
		NeighborList<int> neighbors, 
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		Coord dP_i(0);
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
				dP_i += dp_ij;
				
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

//		dPos[pId] = dP_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacement(
		DeviceArray<Coord> dPos,
		DeviceArray<Real> lambdas,
		DeviceArray<Coord> posArr,
		DeviceArray<Real> massInvArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
				Coord dp_ji = -dp_ij * massInvArr[j];
				dp_ij = dp_ij * massInvArr[pId];
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], dp_ji[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], dp_ji[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], dp_ji[2]);
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dPos, 
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dPos[pId];
	}


	template<typename TDataType>
	DensityPBD<TDataType>::DensityPBD()
		: ConstraintModule()
	{
		this->varIterationNumber()->setValue(3);

		this->varSamplingDistance()->setValue(Real(0.005));
		this->varSmoothingLength()->setValue(Real(0.011));
		this->varRestDensity()->setValue(Real(1000));

		m_summation = std::make_shared<SummationDensity<TDataType>>();

		this->varRestDensity()->connect(m_summation->varRestDensity());
		this->varSmoothingLength()->connect(m_summation->varSmoothingLength());
		this->varSamplingDistance()->connect(m_summation->varSamplingDistance());

		this->inPosition()->connect(m_summation->inPosition());
		this->inNeighborIndex()->connect(m_summation->inNeighborIndex());

		m_summation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	DensityPBD<TDataType>::~DensityPBD()
	{
		m_lamda.release();
		m_deltaPos.release();
		m_position_old.release();
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::constrain()
	{
		int num = this->inPosition()->getElementCount();
		
		if (m_position_old.size() != this->inPosition()->getElementCount())
			m_position_old.resize(this->inPosition()->getElementCount());

		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		if (this->outDensity()->getElementCount() != this->inPosition()->getElementCount())
			this->outDensity()->setElementCount(this->inPosition()->getElementCount());

		if (m_deltaPos.size() != this->inPosition()->getElementCount())
			m_deltaPos.resize(this->inPosition()->getElementCount());

		if (m_lamda.size() != this->inPosition()->getElementCount())
			m_lamda.resize(this->inPosition()->getElementCount());

		int it = 0;

		int itNum = this->varIterationNumber()->getValue();
		while (it < itNum)
		{
			takeOneIteration();

			it++;
		}

		updateVelocity();

		return true;
	}


	template<typename TDataType>
	void DensityPBD<TDataType>::takeOneIteration()
	{
		Real dt = this->getParent()->getDt();

		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		

		m_deltaPos.reset();

		m_summation->update();

		if (m_massInv.isEmpty())
		{
			cuExecute(num, K_ComputeLambdas,
				m_lamda,
				m_summation->outDensity()->getValue(),
				this->inPosition()->getValue(),
				this->inNeighborIndex()->getValue(),
				m_kernel,
				this->varSmoothingLength()->getValue());

			cuExecute(num, K_ComputeDisplacement,
				m_deltaPos,
				m_lamda,
				this->inPosition()->getValue(),
				this->inNeighborIndex()->getValue(),
				m_kernel,
				this->varSmoothingLength()->getValue(),
				dt);
		}
		else
		{
			cuExecute(num, K_ComputeLambdas,
				m_lamda,
				m_summation->outDensity()->getValue(),
				this->inPosition()->getValue(),
				m_massInv.getValue(),
				this->inNeighborIndex()->getValue(),
				m_kernel,
				this->varSmoothingLength()->getValue());

			cuExecute(num, K_ComputeDisplacement,
				m_deltaPos,
				m_lamda,
				this->inPosition()->getValue(),
				m_massInv.getValue(),
				this->inNeighborIndex()->getValue(),
				m_kernel,
				this->varSmoothingLength()->getValue(),
				dt);
		}

		cuExecute(num, K_UpdatePosition,
			this->inPosition()->getValue(),
			this->inVelocity()->getValue(),
			m_deltaPos,
			dt);
	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocity(
		DeviceArray<Coord> velArr,
		DeviceArray<Coord> prePos,
		DeviceArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	template<typename TDataType>
	void DensityPBD<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->getElementCount();

		Real dt = this->getParent()->getDt();

		cuExecute(num, DP_UpdateVelocity,
			this->inVelocity()->getValue(),
			m_position_old,
			this->inPosition()->getValue(),
			dt);
	}

#ifdef PRECISION_FLOAT
	template class DensityPBD<DataType3f>;
#else
 	template class DensityPBD<DataType3d>;
#endif
}