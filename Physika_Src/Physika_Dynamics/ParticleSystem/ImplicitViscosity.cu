#include <cuda_runtime.h>
#include "ImplicitViscosity.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika
{
	template<typename Real>
	__device__ Real VB_VisWeight(const Real r, const Real h)
	{
		Real q = r / h;
		if (q > 1.0f) return 0.0;
		else {
			const Real d = 1.0f - q;
			const Real RR = h*h;
			return 45.0f / (13.0f * (Real)M_PI * RR *h) *d;
		}
	}

	template<typename Real, typename Coord>
	__global__ void K_ApplyViscosity(
		DeviceArray<Coord> velNew,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		DeviceArray<Coord> velOld,
		DeviceArray<Coord> velArr,
		Real viscosity,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real r;
		Coord dv_i(0);
		Coord pos_i = posArr[pId];
		Coord vel_i = velArr[pId];
		Real totalWeight = 0.0f;
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = VB_VisWeight(r, smoothingLength);
				totalWeight += weight;
				dv_i += weight * velArr[j];
			}
		}

		Real b = dt*viscosity / smoothingLength;

		b = totalWeight < EPSILON ? 0.0f : b;

		totalWeight = totalWeight < EPSILON ? 1.0f : totalWeight;

		dv_i /= totalWeight;

		velNew[pId] = velOld[pId] / (1.0f + b) + dv_i*b / (1.0f + b);
	}

	template<typename Real, typename Coord>
	__global__ void VB_UpdateVelocity(
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dVel)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] = dVel[pId];
	}

	template<typename TDataType>
	ImplicitViscosity<TDataType>::ImplicitViscosity()
		:ConstraintModule()
		, m_neighborhoodID(MechanicalState::particle_neighbors())
		, m_smoothingLength(0.0125)
		, m_maxInteration(5)
	{
		m_viscosity = HostVarField<Real>::createField(this, "viscosity", "Viscosity", Real(0.05));
	}

	template<typename TDataType>
	ImplicitViscosity<TDataType>::~ImplicitViscosity()
	{
		m_velOld.release();
		m_velBuf.release();
	}

	template<typename TDataType>
	bool ImplicitViscosity<TDataType>::constrain()
	{
		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for ImplicitViscosity!" << std::endl;
			return false;
		}

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborhoodID);

		if (posFd == nullptr || velFd == nullptr || neighborFd == nullptr)
		{
			std::cout << "Incomplete inputs for ImplicitViscosity!" << std::endl;
			return false;
		}

		int num = posFd->size();

		if (m_velOld.size() != num)
		{
			m_velOld.resize(num);
		}
		if (m_velBuf.size() != num)
		{
			m_velBuf.resize(num);
		}

		cuint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real vis = m_viscosity->getValue();
		Real dt = getParent()->getDt();
		Function1Pt::copy(m_velOld, velFd->getValue());
		for (int t = 0; t < m_maxInteration; t++)
		{
			Function1Pt::copy(m_velBuf, velFd->getValue());
			K_ApplyViscosity << < pDims, BLOCK_SIZE >> > (
				velFd->getValue(), 
				posFd->getValue(),
				neighborFd->getValue(),
				m_velOld, 
				m_velBuf, 
				vis,
				m_smoothingLength, 
				dt);
		}

		return true;
	}

}