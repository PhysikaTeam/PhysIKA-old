//#include "Core/Utilities/template_functions.h"
#include "Helmholtz.h"
#include "Framework/Framework/Node.h"
#include <string>
#include "Kernel.h"
#include "SummationDensity.h"
#include "Core/Utility.h"
#include <thrust/reduce.h>

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(Helmholtz, TDataType)

	template<typename Real>
	__device__ inline Real ExpWeight(const Real r, const Real h)
	{
		const Real q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const Real d = 1.0 - q;
			const Real hh = h*h;
			return d*d*d;// (1.0 - q*q*q*q);
		}

// 		const Real q = r / h;
// 		if (q > 1.0f) return 0.0f;
// 		else {
// 			const Real d = 1.0f - q;
// 			const Real hh = h*h;
// 			//			return 45.0f / ((float)M_PI * hh*h) *d*d;
// 			return (1.0 - q*q*q*q)*h*h;
// 		}
// 		Real q = r / h;
// 		if (q > Real(1))
// 		{
// 			return Real(0);
// 		}
// 		return pow(Real(M_E), -q*q / 2);
	}

	template<typename Real>
	__device__ inline Real ExpWeightGradient(const Real r, const Real h)
	{
		const Real q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const Real d = 1.0 - q;
			const Real hh = h*h;
			return 3*d*d / h;// (1.0 - q*q*q*q);
		}
	}

	template<typename Real>
	__device__ inline Real ExpWeightLaplacian(const Real r, const Real h)
	{
		const Real q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const Real d = 1.0 - q;
			const Real hh = h*h;
			return 6 * d / hh;// (1.0 - q*q*q*q);
		}
	}

	template<typename Real>
	__device__ inline Real kernSpikey(const Real r, const Real h)
	{
		const Real q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const Real d = 1.0 - q;
			const Real hh = h*h;
			return d*d*d;// (1.0 - q*q*q*q);
		}
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
		, m_referenceRho(1000)
		, m_scale(1)
		, m_lambda(0.1)
		, m_kappa(0.0)
	{
		m_bSetup = false;
	}

	template<typename TDataType>
	Helmholtz<TDataType>::~Helmholtz()
	{
		m_bufPos.release();
		m_originPos.release();
	}

	template<typename TDataType>
	bool Helmholtz<TDataType>::initializeImpl()
	{
		return true;
	}

	template<typename Real>
	__device__ Real H_ComputeBulkEnergyGradient(
		Real rho,
		Real restRho,
		Real lambda)
	{
		Real ratio = rho / restRho;
		ratio = ratio < Real(1) ? Real(1) : ratio;
		return lambda*(ratio*ratio - 1)*ratio / restRho;
	}

	template <typename Real, typename Coord>
	__global__ void H_ComputeEnergy(
		DeviceArray<Real> energy,
		DeviceArray<Coord> curPos,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		Coord curPos_i = curPos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		Coord grad_i = Coord(0);
		Real totalWeight = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord curPos_j = curPos[j];
			Real r_ij = (curPos_i - curPos_j).norm();
			if (r_ij > EPSILON)
			{
				Real sw_ij = scale*ExpWeightGradient(r_ij, smoothingLength);
				grad_i += sw_ij*(curPos_j - curPos_i) / r_ij;
				totalWeight += sw_ij;
			}
		}

		if (totalWeight > EPSILON)
		{
			grad_i /= totalWeight;
		}

		energy[pId] = grad_i.dot(grad_i);
	}

	template <typename Real, typename Coord>
	__global__ void H_TakeOneIteration(
		DeviceArray<Coord> newPos,
		DeviceArray<Coord> curPos,
		DeviceArray<Coord> prePos,
		DeviceArray<Real> c,
		DeviceArray<Real> lc,
		DeviceArray<Real> energy,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real restRho,
		Real lambda,
		Real kappa,
		Real scale,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= newPos.size()) return;

		Real BE_i = H_ComputeBulkEnergyGradient(c[pId], restRho, lambda);
		Real lc_i = lc[pId];
		Coord curPos_i = curPos[pId];

		Real factor = dt*dt / restRho / (smoothingLength*smoothingLength);

		Real a_i = Real(0);
		Coord accPos_j = Coord(0);
		Coord accPos_i = Coord(0);
		Coord accPos_ij = Coord(0);
		Coord expPos_i = Coord(0);
		Coord grad_i = Coord(0);
		Real totalWeight = Real(0);
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord curPos_j = curPos[j];
			Real BE_j = H_ComputeBulkEnergyGradient(c[j], restRho, lambda);
			Real lc_j = lc[j];
			Real r_ij = (curPos_i - curPos_j).norm();
			Real R_ij = r_ij < 0.5*smoothingLength ? 0.5*smoothingLength : r_ij;
			if (r_ij > EPSILON)
			{
				Real w_ij = scale*ExpWeightGradient(r_ij, smoothingLength) / r_ij;

				Real sw_ij = scale*ExpWeightGradient(r_ij, smoothingLength);
				grad_i += sw_ij*(energy[pId])*(curPos_j - curPos_i) / r_ij;
				totalWeight += sw_ij;

				Real max_lc = 4*pow(Real(10), Real(8));

				Real fe_ij = -((BE_i + BE_j) / 2 - 0.000000000001*(max_lc + max_lc))*w_ij;

				accPos_ij += fe_ij*(curPos_j-curPos_i);
				accPos_j += fe_ij*curPos_j;
				accPos_i += fe_ij*curPos_i;
				expPos_i += fe_ij*(curPos_j - curPos_i);

				a_i += fe_ij;
			}
		}

		
// 		if (totalWeight > EPSILON)
// 		{
// 			grad_i /= totalWeight;
// 		}
// 
// 		printf("%f \n", totalWeight);
// 
// 		Real energy_i = grad_i.dot(grad_i);
// 		if (grad_i.norm() > EPSILON)
// 		{
// 			grad_i.normalize();
// 		}


		//newPos[pId] = curPos_i + 0.000000000005f*expPos_i;
//		newPos[pId] = curPos_i + 0.00001*expPos_i;

//		newPos[pId] = curPos_i + 0.1*factor*accPos_j - 0.1*factor*accPos_i + 0.0000000001*grad_i;

		newPos[pId] = curPos_i + 0.1*factor*accPos_j - 0.1*factor*accPos_i;

//		newPos[pId] = curPos_i + 0.1*factor*accPos_j - 0.1*factor*accPos_i +0.0001*(1-lc_i/(5*pow(Real(10), Real(8))))*grad_i;

//		printf("%f \n", (1 - lc_i / (5 * pow(Real(10), Real(8)))));

// 		Real cr = Real(1);
// 		if (0.1*factor*a_i < - 1.0/3.0)
// 		{
// 			cr *= -0.1*factor*a_i / 3;
// 		}

//		newPos[pId] = (prePos[pId] + 0.1*factor*accPos_j) / (1 + 0.1*factor*a_i);
// 
// 		if(0.1*factor*a_i < -0.5)
// 			printf("%f \n", 0.1*factor*a_i);

//		Coord tmpCord = accPos_i / a_i;

// 		if (c[pId] > 1000 && pId == 5)
// 		{
// 			printf("NewPos: %f %f %f CurPos: %f %f %f TmpPos: %f %f %f PrePos: %f %f %f : %f \n", newPos[pId][0], newPos[pId][1], newPos[pId][2], curPos[pId][0], curPos[pId][1], curPos[pId][2], tmpCord[0], tmpCord[1], tmpCord[2], prePos[pId][0], prePos[pId][1], prePos[pId][2], factor*a_i);
// 		}
		
	}

	template <typename Coord>
	__global__ void H_UpdateVelocity(
		DeviceArray<Coord> curVel,
		DeviceArray<Coord> curPos,
		DeviceArray<Coord> oriPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curVel.size()) return;

		curVel[pId] += (curPos[pId]-oriPos[pId])/dt;
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

		int num = posFd->getReference()->size();

		if (m_bufPos.size() != num)
			m_bufPos.resize(num);
		if (m_originPos.size() != num)
			m_originPos.resize(num);


		Real max_c = 0;
		if (!m_bSetup)
		{
			m_c.resize(num);
			m_lc.resize(num);
			m_energy.resize(num);

			computeC(m_c, posFd->getValue(), neighborFd->getValue());
			max_c = thrust::reduce(thrust::device, m_c.getDataPtr(), m_c.getDataPtr() + m_c.size(), (Real)0, thrust::maximum<Real>());
			m_scale = m_referenceRho / max_c;

			m_bSetup = true;
		}
		

		Function1Pt::copy(m_originPos, posFd->getValue());

		Real dt = getParent()->getDt();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		int it = 0;
		while (it < 5)
		{
			//printf("Iteration %d: \n", it);

			computeC(m_c, posFd->getValue(), neighborFd->getValue());
			computeLC(m_lc, posFd->getValue(), neighborFd->getValue());
			
//			Real max_lc = thrust::reduce(thrust::device, m_lc.getDataPtr(), m_lc.getDataPtr() + m_lc.size(), (Real)0, thrust::maximum<Real>());

//			printf("%f \n", max_lc);

			Function1Pt::copy(m_bufPos, posFd->getValue());

			H_ComputeEnergy << <pDims, BLOCK_SIZE >> > (
				m_energy,
				posFd->getValue(),
				neighborFd->getValue(),
				m_smoothingLength,
				m_scale);

			H_TakeOneIteration << <pDims, BLOCK_SIZE >> > (
				posFd->getValue(),
				m_bufPos,
				m_originPos,
				m_c,
				m_lc,
				m_energy,
				neighborFd->getValue(),
				m_smoothingLength,
				m_referenceRho,
				m_lambda,
				m_kappa,
				m_scale,
				dt);
			it++;
		}

		H_UpdateVelocity << <pDims, BLOCK_SIZE >> > (velFd->getValue(), posFd->getValue(), m_originPos, dt);

		return true;
	}

	template <typename Real, typename Coord>
	__global__ void H_ComputeC(
		DeviceArray<Real> c,
		DeviceArray<Coord> pos,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		SpikyKernel<Real> kern;

		Real r;
		Real c_i = Real(0);
		Coord pos_i = pos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - pos[j]).norm();
			c_i += scale*ExpWeight(r, smoothingLength);
		}
		c[pId] = c_i;
	}

	template<typename TDataType>
	void Helmholtz<TDataType>::computeC(DeviceArray<Real>& c, DeviceArray<Coord>& pos, NeighborList<int>& neighbors)
	{
		uint pDims = cudaGridSize(c.size(), BLOCK_SIZE);
		H_ComputeC << <pDims, BLOCK_SIZE >> > (c, pos, neighbors, m_smoothingLength, m_scale);
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
		Real smoothingLength,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Real term = Real(0);
		Coord pos_i = pos[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - pos[j]).norm();
			if (r > EPSILON)
			{
				Real w = scale*ExpWeightLaplacian(r, smoothingLength);
				term += w;
			}
		}
		lc[pId] = term;
	}

	template<typename TDataType>
	void Helmholtz<TDataType>::computeLC(DeviceArray<Real>& lc, DeviceArray<Coord>& pos, NeighborList<int>& neighbors)
	{
		uint pDims = cudaGridSize(m_c.size(), BLOCK_SIZE);
		H_ComputeLC << <pDims, BLOCK_SIZE >> > (lc, pos, neighbors, m_smoothingLength, m_scale);
	}
}