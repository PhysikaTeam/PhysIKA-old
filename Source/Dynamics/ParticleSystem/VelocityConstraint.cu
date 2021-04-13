#include <cuda_runtime.h>
#include "VelocityConstraint.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "SummationDensity.h"
#include "Attribute.h"
#include "Kernel.h"

namespace PhysIKA
{
	__device__ inline float kernWeight(const float r, const float h)
	{
		const float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const float d = 1.0f - q;
			const float hh = h*h;
//			return 45.0f / ((float)M_PI * hh*h) *d*d;
			return (1.0-pow(q, 4.0f));
//			return (1.0 - q)*(1.0 - q)*h*h;
		}
	}

	__device__ inline float kernWR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.4f*h);
		}
		return w / r;
	}

	__device__ inline float kernWRR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.16f*h*h);
		}
		return w / r/r;
	}


	template <typename Real, typename Coord>
	__global__ void VC_ComputeAlpha
	(
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real alpha_i = 0.0f;

		int nbSize = neighbors.getNeighborSize(pId);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();;

			if (r > EPSILON)
			{
				Real a_ij = kernWeight(r, smoothingLength);
				alpha_i += a_ij;
			}
		}

		alpha[pId] = alpha_i;
	}

	template <typename Real>
	__global__ void VC_CorrectAlpha
	(
		DeviceArray<Real> alpha,
		Real maxAlpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= alpha.size()) return;

		Real alpha_i = alpha[pId];
		if (alpha_i < maxAlpha)
		{
			alpha_i = maxAlpha;
		}
		alpha[pId] = alpha_i;
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeDiagonalElement
	(
		DeviceArray<Real> AiiFluid,
		DeviceArray<Real> AiiTotal,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Real invAlpha = 1.0f / alpha[pId];


		Real diaA_total = 0.0f;
		Real diaA_fluid = 0.0f;
		Coord pos_i = position[pId];

		bool bNearWall = false;
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			Attribute att_j = attribute[j];
			if (r > EPSILON)
			{
				Real wrr_ij = invAlpha*kernWRR(r, smoothingLength);
				if (att_j.IsDynamic())
				{
					diaA_total += wrr_ij;
					diaA_fluid += wrr_ij;
					atomicAdd(&AiiFluid[j], wrr_ij);
					atomicAdd(&AiiTotal[j], wrr_ij);
				}
				else
				{
					diaA_total += 2.0f*wrr_ij;
				}
			}
		}

		atomicAdd(&AiiFluid[pId], diaA_fluid);
		atomicAdd(&AiiTotal[pId], diaA_total);
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeDiagonalElement
	(
		DeviceArray<Real> diaA,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];
		Real A_i = 0.0f;

		int nbSize = neighbors.getNeighborSize(pId);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].IsDynamic())
			{
				Real wrr_ij = invAlpha_i*kernWRR(r, smoothingLength);
				A_i += wrr_ij;
				atomicAdd(&diaA[j], wrr_ij);
			}
		}

		atomicAdd(&diaA[pId], A_i);
	}

	template <typename Real, typename Coord>
	__global__ void VC_DetectSurface
	(
		DeviceArray<Real> Aii,
		DeviceArray<bool> bSurface,
		DeviceArray<Real> AiiFluid,
		DeviceArray<Real> AiiTotal,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real maxA
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Real total_weight = 0.0f;
		Coord div_i = Coord(0);

		SmoothKernel<Real> kernSmooth;

		Coord pos_i = position[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		bool bNearWall = false;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].IsDynamic())
			{
				float weight = -kernSmooth.Gradient(r, smoothingLength);
				total_weight += weight;
				div_i += (position[j] - pos_i)*(weight / r);
			}

			if (!attribute[j].IsDynamic())
			{
				bNearWall = true;
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDiv = div_i.norm() / total_weight;

		bool bSurface_i = false;
		Real diagF_i = AiiFluid[pId];
		Real diagT_i = AiiTotal[pId];
		Real aii = diagF_i;
		Real eps = 0.001f;
		Real diagS_i = diagT_i - diagF_i;
		Real threshold = 0.0f;
		if (bNearWall && diagT_i < maxA*(1.0f - threshold))
		{
			bSurface_i = true;
			aii = maxA - (diagT_i - diagF_i);
		}

		if (!bNearWall && diagF_i < maxA*(1.0f - threshold))
		{
			bSurface_i = true;
			aii = maxA;
		}
		bSurface[pId] = bSurface_i;
		Aii[pId] = aii;
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeDivergence
	(
		DeviceArray<Real> divergence,
		DeviceArray<Real> alpha,
		DeviceArray<Real> density,
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		DeviceArray<bool> bSurface,
		DeviceArray<Coord> normals,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real separation,
		Real tangential,
		Real restDensity,
		Real smoothingLength,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Coord vel_i = velocity[pId];

		Real div_vi = 0.0f;

		Real invAlpha_i = 1.0f / alpha[pId];

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].IsFluid())
			{
				Real wr_ij = kernWR(r, smoothingLength);
				Coord g = -invAlpha_i*(pos_i - position[j])*wr_ij*(1.0f / r);

				if (attribute[j].IsDynamic())
				{
					Real div_ij = 0.5f*(vel_i - velocity[j]).dot(g)*restDensity / dt;	//dv_ij = 1 / alpha_i * (v_i-v_j).*(x_i-x_j) / r * (w / r);
					atomicAdd(&divergence[pId], div_ij);
					atomicAdd(&divergence[j], div_ij);
				}
				else
				{
					//float div_ij = dot(2.0f*vel_i, g)*const_vc_state.restDensity / dt;
					Coord normal_j = normals[j];

					Coord dVel = vel_i - velocity[j];
					Real magNVel = dVel.dot(normal_j);
					Coord nVel = magNVel*normal_j;
					Coord tVel = dVel - nVel;

					//float div_ij = dot(2.0f*dot(vel_i - velArr[j], normal_i)*normal_i, g)*const_vc_state.restDensity / dt;
					//printf("Boundary dVel: %f %f %f\n", div_ij, pos_i.x, pos_i.y);
					//printf("Normal: %f %f %f; Position: %f %f %f \n", normal_i.x, normal_i.y, normal_i.z, posArr[j].x, posArr[j].y, posArr[j].z);
					if (magNVel < -EPSILON)
					{
						Real div_ij = g.dot(2.0f*(nVel + tangential*tVel))*restDensity / dt;
						//						printf("Boundary div: %f \n", div_ij);
						atomicAdd(&divergence[pId], div_ij);
					}
					else
					{
						Real div_ij = g.dot(2.0f*(separation*nVel + tangential*tVel))*restDensity / dt;
						atomicAdd(&divergence[pId], div_ij);
					}

				}
			}
		}
		// 		if (rhoArr[pId] > const_vc_state.restDensity)
		// 		{
		// 			atomicAdd(&divArr[pId], 1000.0f/const_vc_state.smoothingLength*(rhoArr[pId] - const_vc_state.restDensity) / (const_vc_state.restDensity * dt));
		// 		}
	}

	template <typename Real, typename Coord>
	__global__ void VC_CompensateSource
	(
		DeviceArray<Real> divergence,
		DeviceArray<Real> density,
		DeviceArray<Attribute> attribute,
		DeviceArray<Coord> position,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= density.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		if (density[pId] > restDensity)
		{
			Real ratio = (density[pId] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 100000.0f*ratio / dt);
		}
	}

	// compute Ax;
	template <typename Real, typename Coord>
	__global__ void VC_ComputeAx
	(
		DeviceArray<Real> residual,
		DeviceArray<Real> pressure,
		DeviceArray<Real> aiiSymArr,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbor,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		atomicAdd(&residual[pId], aiiSymArr[pId] * pressure[pId]);
		Real con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		int nbSize = neighbor.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].IsDynamic())
			{
				Real wrr_ij = kernWRR(r, smoothingLength);
				Real a_ij = -invAlpha_i*wrr_ij;
				//				residual += con1*a_ij*preArr[j];
				atomicAdd(&residual[pId], con1*a_ij*pressure[j]);
				atomicAdd(&residual[j], con1*a_ij*pressure[pId]);
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void VC_UpdateVelocityBoundaryCorrected(
		DeviceArray<Real> pressure,
		DeviceArray<Real> alpha,
		DeviceArray<bool> bSurface,
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> normal,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbor,
		Real restDensity,
		Real airPressure,
		Real sliding,
		Real separation,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (attribute[pId].IsDynamic())
		{
			Coord pos_i = position[pId];
			Real p_i = pressure[pId];

			int nbSize = neighbor.getNeighborSize(pId);

			Real total_weight = 0.0f;

			Real ceo = 1.6f;

			Real invAlpha = 1.0f / alpha[pId];
			Coord vel_i = velocity[pId];
			Coord dv_i(0.0f);
			Real scale = 1.0f*dt / restDensity;
			Real acuP = 0.0f;
			total_weight = 0.0f;
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = neighbor.getElement(pId, ne);
				Real r = (pos_i - position[j]).norm();

				Attribute att_j = attribute[j];
				if (r > EPSILON)
				{
					Real weight = -invAlpha*kernWR(r, smoothingLength);
					Coord dnij = (pos_i - position[j])*(1.0f / r);
					Coord corrected = dnij;
					if (corrected.norm() > EPSILON)
					{
						corrected = corrected.normalize();
					}
					corrected = -scale*weight*corrected;

					Coord dvij = (pressure[j] - pressure[pId])*corrected;
					Coord dvjj = (pressure[j] + airPressure) * corrected;
					Coord dvij_sym = 0.5f*(pressure[pId] + pressure[j])*corrected;

					//Calculate asymmetric pressure force
					if (att_j.IsDynamic())
					{
						if (bSurface[pId])
						{
							dv_i += dvjj;
						}
						else
						{
							dv_i += dvij;
						}

						if (bSurface[j])
						{
							Coord dvii = -(pressure[pId] + airPressure) * corrected;
							atomicAdd(&velocity[j][0], ceo*dvii[0]);
							atomicAdd(&velocity[j][1], ceo*dvii[1]);
							atomicAdd(&velocity[j][2], ceo*dvii[2]);
						}
						else
						{
							atomicAdd(&velocity[j][0], ceo*dvij[0]);
							atomicAdd(&velocity[j][1], ceo*dvij[1]);
							atomicAdd(&velocity[j][2], ceo*dvij[2]);
						}
					}
					else
					{
						Coord dvii = 2.0f*(pressure[pId]) * corrected;
						if (bSurface[pId])
						{
							dv_i += dvii;
						}

						float weight = 2.0f*invAlpha*kernWeight(r, smoothingLength);
						Coord nij = (pos_i - position[j]);
						if (nij.norm() > EPSILON)
						{
							nij = nij.normalize();
						}
						else
							nij = Coord(1.0f, 0.0f, 0.0f);

						Coord normal_j = normal[j];
						Coord dVel = velocity[j] - vel_i;
						Real magNVel = dVel.dot(normal_j);
						Coord nVel = magNVel*normal_j;
						Coord tVel = dVel - nVel;
						if (magNVel > EPSILON)
						{
							dv_i += weight*nij.dot(nVel + sliding*tVel)*nij;
						}
						else
						{
							dv_i += weight*nij.dot(separation*nVel + sliding*tVel)*nij;
						}


						// 						float weight = 2.0f*invAlpha*kernWRR(r, const_vc_state.smoothingLength);
						// 						float3 nij = (pos_i - posArr[j]);
						// 						//printf("Normal: %f %f %f; Position: %f %f %f \n", normal_i.x, normal_i.y, normal_i.z, posArr[j].x, posArr[j].y, posArr[j].z);
						// 						dv_i += weight*dot(velArr[j] - vel_i, nij)*nij;
					}

				}
			}

			dv_i *= ceo;

			atomicAdd(&velocity[pId][0], dv_i[0]);
			atomicAdd(&velocity[pId][1], dv_i[1]);
			atomicAdd(&velocity[pId][2], dv_i[2]);
		}
	}

	template<typename TDataType>
	VelocityConstraint<TDataType>::VelocityConstraint()
		: ConstraintModule()
		, m_airPressure(Real(0))
		, m_reduce(NULL)
		, m_arithmetic(NULL)
	{
		m_smoothingLength.setValue(Real(0.011));

		attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);

		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_normal, "normal", "Storing the particle normals!", false);
		attachField(&m_attribute, "attribute", "Storing the particle attributes!", false);
		attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
	}

	template<typename TDataType>
	VelocityConstraint<TDataType>::~VelocityConstraint()
	{
		m_alpha.release();
		m_Aii.release();
		m_AiiFluid.release();
		m_AiiTotal.release();
		m_pressure.release();
		m_divergence.release();
		m_bSurface.release();

		m_y.release();
		m_r.release();
		m_p.release();

		m_pressure.release();

		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
	}

	template<typename TDataType>
	bool VelocityConstraint<TDataType>::constrain()
	{
		Real dt = getParent()->getDt();

		uint pDims = cudaGridSize(m_position.getElementCount(), BLOCK_SIZE);

		//compute alpha_i = sigma w_j and A_i = sigma w_ij / r_ij / r_ij
		m_alpha.reset();
		VC_ComputeAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha, 
			m_position.getValue(), 
			m_attribute.getValue(), 
			m_neighborhood.getValue(), 
			m_smoothingLength.getValue());
		VC_CorrectAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha, 
			m_maxAlpha);

		//compute the diagonal elements of the coefficient matrix
		m_AiiFluid.reset();
		m_AiiTotal.reset();
		VC_ComputeDiagonalElement << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid, 
			m_AiiTotal, 
			m_alpha, 
			m_position.getValue(),
			m_attribute.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue());

		m_bSurface.reset();
		m_Aii.reset();
		VC_DetectSurface << <pDims, BLOCK_SIZE >> > (
			m_Aii, 
			m_bSurface, 
			m_AiiFluid,
			m_AiiTotal,
			m_position.getValue(),
			m_attribute.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue(),
			m_maxA);

		int itor = 0;

		//compute the source term
		m_densitySum->compute();
		m_divergence.reset();
		VC_ComputeDivergence << <pDims, BLOCK_SIZE >> > (
			m_divergence, 
			m_alpha, 
			m_densitySum->outDensity()->getValue(),
			m_position.getValue(), 
			m_velocity.getValue(), 
			m_bSurface, 
			m_normal.getValue(), 
			m_attribute.getValue(), 
			m_neighborhood.getValue(), 
			m_separation, 
			m_tangential, 
			m_restDensity,
			m_smoothingLength.getValue(), 
			dt);
		VC_CompensateSource << <pDims, BLOCK_SIZE >> > (
			m_divergence, 
			m_density, 
			m_attribute.getValue(), 
			m_position.getValue(), 
			m_restDensity, 
			dt);
		
		//solve the linear system of equations with a conjugate gradient method.
		m_y.reset();
		VC_ComputeAx << <pDims, BLOCK_SIZE >> > (
			m_y, 
			m_pressure, 
			m_Aii, 
			m_alpha, 
			m_position.getValue(),
			m_attribute.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue());

		m_r.reset();
		Function2Pt::subtract(m_r, m_divergence, m_y);
		Function1Pt::copy(m_p, m_r);
		Real rr = m_arithmetic->Dot(m_r, m_r);
		Real err = sqrt(rr / m_r.size());

		while (itor < 1000 && err > 1.0f)
		{
			m_y.reset();
			//VC_ComputeAx << <pDims, BLOCK_SIZE >> > (*yArr, *pArr, *aiiArr, *alphaArr, *posArr, *attArr, *neighborArr);
			VC_ComputeAx << <pDims, BLOCK_SIZE >> > (
				m_y, 
				m_p, 
				m_Aii, 
				m_alpha, 
				m_position.getValue(),
				m_attribute.getValue(),
				m_neighborhood.getValue(),
				m_smoothingLength.getValue());

			float alpha = rr / m_arithmetic->Dot(m_p, m_y);
			Function2Pt::saxpy(m_pressure, m_p, m_pressure, alpha);
			Function2Pt::saxpy(m_r, m_y, m_r, -alpha);

			Real rr_old = rr;

			rr = m_arithmetic->Dot(m_r, m_r);

			Real beta = rr / rr_old;
			Function2Pt::saxpy(m_p, m_p, m_r, beta);

			err = sqrt(rr / m_r.size());

			itor++;
		}

		//update the each particle's velocity
		VC_UpdateVelocityBoundaryCorrected << <pDims, BLOCK_SIZE >> > (
			m_pressure,
			m_alpha,
			m_bSurface, 
			m_position.getValue(), 
			m_velocity.getValue(), 
			m_normal.getValue(), 
			m_attribute.getValue(), 
			m_neighborhood.getValue(),
			m_restDensity,
			m_airPressure,
			m_tangential,
			m_separation,
			m_smoothingLength.getValue(),
			dt);

		return true;
	}

	template<typename TDataType>
	bool VelocityConstraint<TDataType>::initializeImpl()
	{
		if (!isAllFieldsReady())
		{
			Log::sendMessage(Log::Error, "VelocityConstraint's fields are not fully initialized!");
			return false;
		}

		m_densitySum = std::make_shared<SummationDensity<TDataType>>();
		m_smoothingLength.connect(m_densitySum->varSmoothingLength());
		m_position.connect(m_densitySum->inPosition());
		m_neighborhood.connect(m_densitySum->inNeighborIndex());
		m_densitySum->initialize();

		int num = m_position.getElementCount();
		m_alpha.resize(num);
		m_Aii.resize(num);
		m_AiiFluid.resize(num);
		m_AiiTotal.resize(num);
		m_pressure.resize(num);
		m_divergence.resize(num);
		m_bSurface.resize(num);
		m_density.resize(num);

		m_y.resize(num);
		m_r.resize(num);
		m_p.resize(num);

		m_pressure.resize(num);

		m_reduce = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num);


		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_alpha.reset();
		VC_ComputeAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_position.getValue(),
			m_attribute.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue());

		m_maxAlpha = m_reduce->maximum(m_alpha.getDataPtr(), m_alpha.size());

		VC_CorrectAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_maxAlpha);

		m_AiiFluid.reset();
		VC_ComputeDiagonalElement << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid,
			m_alpha,
			m_position.getValue(),
			m_attribute.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue());

		m_maxA = m_reduce->maximum(m_AiiFluid.getDataPtr(), m_AiiFluid.size());

		std::cout << "Max alpha: " << m_maxAlpha << std::endl;
		std::cout << "Max A: " << m_maxA << std::endl;

		return true;
	}

}