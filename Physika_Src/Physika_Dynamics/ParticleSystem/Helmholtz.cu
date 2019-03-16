#include <cuda_runtime.h>
//#include "Physika_Core/Utilities/template_functions.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Helmholtz.h"
#include "Physika_Framework/Framework/Node.h"
#include <string>
#include "Kernel.h"
#include "DensitySummation.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(Helmholtz, TDataType)

	template<typename Real>
	__device__ inline Real ExpWeight(const Real r, const Real h)
	{
		Real q = r / h;
		return pow(Real(M_E), -q*q / 2);
	}

		__device__ inline float kernWeight1(const float r, const float h)
	{
		const float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const float d = 1.0f - q;
			const float hh = h*h;
			//			return 45.0f / ((float)M_PI * hh*h) *d*d;
			return (1.0 - q*q*q*q)*h*h;
		}
	}

	__device__ inline float kernWR1(const float r, const float h)
	{
		float w = kernWeight1(r, h);
		const float q = r / h;
		if (q < 0.5f)
		{
			return w / (0.5f*h);
		}
		return w / r;
	}

	__device__ inline float kernWRR1(const float r, const float h)
	{
		float w = kernWeight1(r, h);
		const float q = r / h;
		if (q < 0.5f)
		{
			return w / (0.25f*h*h);
		}
		return w / r / r;
	}

	template <typename Real, typename Coord>
		__global__ void H_ComputeGradient(
			DeviceArray<Coord> grads,
			DeviceArray<Real> rhoArr,
			DeviceArray<Coord> curPos,
			DeviceArray<Coord> originPos,
			NeighborList<int> neighbors,
			Real bulk,
			Real surfaceTension,
			Real inertia)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		Real a1 = inertia;
		Real a2 = bulk;
		Real a3 = surfaceTension;

		SpikyKernel<Real> kern;

		Real w1 = 1.0f*a1;
		Real w2 = 0.005f*(rhoArr[pId] - 1000.0f) / (1000.0f)*a2;
		if (w2 < EPSILON)
		{
			w2 = 0.0f;
		}
		Real w3 = 0.005f*a3;

		Real mass = 1.0;
		Real h = 0.0125f;

		Coord pos_i = curPos[pId];

		Real lamda_i = 0.0f;
		Coord grad1_i = originPos[pId] - pos_i;

		Coord grad2 = Coord(0);
		Real total_weight2 = 0.0f;
		Coord grad3 = Coord(0);
		Real total_weight3 = 0.0f;

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord pos_j = curPos[j];
			Real r = (pos_i - pos_j).norm();

			if (r > EPSILON)
			{
				Real weight2 = -mass*kern.Gradient(r, h);
				total_weight2 += weight2;
				Coord g2_ij = weight2*(pos_i - pos_j) * (1.0f / r);
				grad2 += g2_ij;

				Real weight3 = kernWRR1(r, h);
				total_weight3 += weight3;
				Coord g3_ij = weight3*(pos_i - pos_j)* (1.0f / r);
				grad3 += g3_ij;
			}
		}

		total_weight2 = total_weight2 < EPSILON ? 1.0f : total_weight2;
		total_weight3 = total_weight3 < EPSILON ? 1.0f : total_weight3;

		grad2 /= total_weight2;
		grad3 /= total_weight3;

		Coord nGrad3;
		if (grad3.norm() > EPSILON)
		{
			nGrad3 = grad3.normalize();
		}

		Real energy = grad3.dot(grad3);
		Real tweight = 0;
		Coord grad4 = Coord(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord pos_j = curPos[j];
			Real r = (pos_i - pos_j).norm();

			if (r > EPSILON)
			{
				float weight2 = -mass*kern.Gradient(r, h);
				Coord g2_ij = (weight2 / total_weight2)*(pos_i - pos_j) * (1.0f / r);
// 				atomicAdd(&grads[j].x, -w2*g2_ij.x);
// 				atomicAdd(&grads[j].y, -w2*g2_ij.y);
// 				atomicAdd(&grads[j].z, -w2*g2_ij.z);
			}
		}

// 		atomicAdd(&grads[pId].x, w1*grad1_i.x + w2*grad2.x - w3*energy*nGrad3.x);
// 		atomicAdd(&grads[pId].y, w1*grad1_i.y + w2*grad2.y - w3*energy*nGrad3.y);
// 		atomicAdd(&grads[pId].z, w1*grad1_i.z + w2*grad2.z - w3*energy*nGrad3.z);
	}

	template <typename Coord>
	__global__ void H_UpdatePosition(
		DeviceArray<Coord> gradients,
		DeviceArray<Coord> curPos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		curPos[pId] += gradients[pId];
	}

	__global__ void H_UpdateVelocity(
		DeviceArray<float3> curVel,
		DeviceArray<float3> curPos,
		DeviceArray<float3> originalPos,
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curVel.size()) return;

		curVel[pId] += 0.0f*(curPos[pId] - originalPos[pId]) / dt;
	}


	template<typename TDataType>
	Helmholtz<TDataType>::Helmholtz()
		: ConstraintModule()
		, m_posID(MechanicalState::position())
		, m_velID(MechanicalState::velocity())
		, m_neighborhoodID(MechanicalState::particle_neighbors())
		, m_smoothingLength(0.0125)
		, m_maxIteration(5)
	{
	}

	template<typename TDataType>
	Helmholtz<TDataType>::~Helmholtz()
	{
		m_lamda.release();
		m_deltaPos.release();
		m_originPos.release();
	}

	template<typename TDataType>
	bool Helmholtz<TDataType>::initializeImpl()
	{
// 		m_densitySum = getParent()->getModule<DensitySummation<TDataType>>();
// 		if (m_densitySum == nullptr)
// 		{
// 			auto summation = std::make_shared<DensitySummation<TDataType>>();
// 			summation->setSmoothingLength(m_smoothingLength);
// 
// 			getParent()->addModule(summation);
// 		}
		return true;
	}

	template<typename TDataType>
	bool Helmholtz<TDataType>::constrain()
	{
		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for Helmholtz!" << std::endl;
			return false;
		}

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborhoodID);

		if (posFd == nullptr || velFd == nullptr || neighborFd == nullptr)
		{
			std::cout << "Incomplete inputs for Helmholtz!" << std::endl;
			return false;
		}

		int num = posFd->size();

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_rho.size() != num)
			m_rho.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		if (m_originPos.size() != num)
			m_originPos.resize(num);

		Real dt = getParent()->getDt();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		float bulk = 1.0f;
		float st = 0.6f;
		float inertia = 0.3f;

		int it = 0;
		while (it < 30)
		{
			m_deltaPos.reset();

			H_ComputeGradient << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				m_rho,
				posFd->getValue(),
				m_originPos,
				neighborFd->getValue(),
				bulk,
				st,
				inertia);
			H_UpdatePosition << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				posFd->getValue());
			it++;
		}

		return true;
	}

	template <typename Real, typename Coord>
	__global__ void H_ComputeC(
		DeviceArray<Real> c,
		DeviceArray<Coord> pos,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Real r;
		Real c_i = Real(0);
		Coord pos_i = pos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - pos[j]).norm();
			c_i += kernWeight1(r, smoothingLength);
		}
		c[pId] = c_i;
	}

	template<typename TDataType>
	void Helmholtz<TDataType>::computeC()
	{
	}

	template <typename Real, typename Coord>
	__global__ void H_ComputeGC(
		DeviceArray<Coord> gc,
		DeviceArray<Coord> pos,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Real r;
		Coord gc_i = Coord(0);
		Coord pos_i = pos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord pos_j = pos[j];
			r = (pos_i - pos_j).norm();

			if (r > EPSILON)
			{
				gc_i += kernWR1(r, smoothingLength) * (pos_i - pos_j) / r;
			}
			
		}
		gc[pId] = gc_i;
	}

	template<typename TDataType>
	void Helmholtz<TDataType>::computeGC()
	{

	}

	template <typename Real, typename Coord>
	__global__ void H_ComputeLC(
		DeviceArray<Real> lc,
		DeviceArray<Coord> pos,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Real h2_inv = 1 / smoothingLength*smoothingLength;
		Real h4_inv = h2_inv*h2_inv;

		Real term1 = Real(0);
		Real term2 = Real(0);
		Coord pos_i = pos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - pos[j]).norm();
			Real r2 = r*r;
			Real w = ExpWeight(r, smoothingLength);
			term1 += w*r2;
			term2 += w;
		}
		lc[pId] = h4_inv*term1 - 3 * term2*h2_inv;
	}

	template<typename TDataType>
	void Helmholtz<TDataType>::computeLC()
	{

	}
}