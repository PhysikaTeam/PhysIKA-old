#include <cuda_runtime.h>
#include "HyperelasticForce.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Algorithm/MatrixFunc.h"
#include "Kernel.h"

namespace Physika
{
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void K_ComputeDeformationTensor(
		DeviceArray<Matrix> matArr,
		NeighborList<NPair> restShapes,
		SmoothKernel<Real> kernSmooth,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= matArr.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = 0.0f;
		Matrix mat_i = Matrix(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			Real r = (rest_i-rest_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);
				Coord q = (rest_j - rest_i)*sqrt(weight);

				mat_i(0, 0) += q[0] * q[0]; mat_i(0, 1) += q[0] * q[1]; mat_i(0, 2) += q[0] * q[2];
				mat_i(1, 0) += q[1] * q[0]; mat_i(1, 1) += q[1] * q[1]; mat_i(1, 2) += q[1] * q[2];
				mat_i(2, 0) += q[2] * q[0]; mat_i(2, 1) += q[2] * q[1]; mat_i(2, 2) += q[2] * q[2];

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
		}

		Matrix R, U, D;
		polarDecomposition(mat_i, R, U, D);

		Real threshold = 0.0001f*smoothingLength;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		mat_i = R.transpose()*U*D*U.transpose();

		matArr[pId] = mat_i;
	}

/*	template <typename Real, typename Coord, typename Matrix, typename RestShape>
	__global__ void EM_RotateRestShape(
		DeviceArray<Coord> posArr,
		DeviceArray<Matrix> matArr,
		DeviceArray<RestShape> restShapes,
		SmoothKernel<Real> kernSmooth,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord rest_i = restShapes[pId].pos[restShapes[pId].idx];
		int size_i = restShapes[pId].size;

		//			cout << i << " " << rids[shape_i.ids[shape_i.idx]] << endl;
		Real total_weight = 0.0f;
		Matrix mat_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			int j = restShapes[pId].ids[ne];
			float r = restShapes[pId].distance[ne];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord p = posArr[j] - posArr[pId];
				//Vector3f q = (shape_i.pos[ne] - rest_i)*(1.0f/r)*weight;
				float3 q = (restShapes[pId].pos[ne] - rest_i)*weight;

				mat_i(0, 0) += p.x * q.x; mat_i(0, 1) += p.x * q.y; mat_i(0, 2) += p.x * q.z;
				mat_i(1, 0) += p.y * q.x; mat_i(1, 1) += p.y * q.y; mat_i(1, 2) += p.y * q.z;
				mat_i(2, 0) += p.z * q.x; mat_i(2, 1) += p.z * q.y; mat_i(2, 2) += p.z * q.z;
				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
			mat_i *= matArr[pId];
		}

		glm::mat3 glmMat3_i;
		glmMat3_i[0][0] = mat_i(0, 0);
		glmMat3_i[0][1] = mat_i(0, 1);
		glmMat3_i[0][2] = mat_i(0, 2);
		glmMat3_i[1][0] = mat_i(1, 0);
		glmMat3_i[1][1] = mat_i(1, 1);
		glmMat3_i[1][2] = mat_i(1, 2);
		glmMat3_i[2][0] = mat_i(2, 0);
		glmMat3_i[2][1] = mat_i(2, 1);
		glmMat3_i[2][2] = mat_i(2, 2);
		glm::mat3 R, U, D;
		PolarDecompositionStable(glmMat3_i, EPSILON, R);

		Matrix matR;
		matR(0, 0) = R[0][0];
		matR(0, 1) = R[0][1];
		matR(0, 2) = R[0][2];
		matR(1, 0) = R[1][0];
		matR(1, 1) = R[1][1];
		matR(1, 2) = R[1][2];
		matR(2, 0) = R[2][0];
		matR(2, 1) = R[2][1];
		matR(2, 2) = R[2][2];

		for (int ne = 0; ne < size_i; ne++)
		{
			int j = restShapes[pId].ids[ne];
			Real r = restShapes[pId].distance[ne];
			if (r > EPSILON)
			{
				Coord v = restShapes[pId].pos[ne] - rest_i;
// 				Coord v3 = Coord(v.x, v.y, v.z);
// 				v3 = matR*v3;
				Coord v3 = matR*v;
				restShapes[pId].pos[ne] = v3 + rest_i;
			}
		}
	}*/

	__device__ Real EM_GetStiffness1(Real r)
	{
		return Real(0.01*(pow(Real(r), Real(3)) - 1));
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void K_ComputeElasticForce(
		DeviceArray<Coord> force,
		DeviceArray<Coord> accuPos,
		DeviceArray<Real> accuLamdas,
		DeviceArray<Real> bulkCoefs,
		DeviceArray<Coord> posArr,
		DeviceArray<Matrix> matArr,
		NeighborList<NPair> restShapes,
		SmoothKernel<Real> kernSmooth,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);
		Coord pos_i = posArr[pId];

		//			cout << i << " " << rids[shape_i.ids[shape_i.idx]] << endl;
		Real total_weight = 0.0f;
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.j;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord p = (posArr[j] - posArr[pId]);
				Coord q = (rest_j - rest_i)*weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
			//deform_i *= matArr[pId];
			deform_i = deform_i * matArr[pId];
		}
		else
		{
			total_weight = 1.0f;
		}

		if ((deform_i.determinant()) < 0.01f)
		{
			deform_i = Matrix::identityMatrix();
		}

		Matrix mat_i = deform_i;

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.j;
			Real r = (rest_j - rest_i).norm();
			Coord pos_j = posArr[j];
			Real l = (pos_j - pos_i).norm();
			
			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord q = (rest_j - rest_i)*(1.0f / r);
				Coord p = mat_i*q;
				Coord p_ij = (pos_j-pos_i);

				p.normalize();
				p_ij.normalize();

				p = 0.7*p_ij -0.5*p;

				Real bulk_ij = EM_GetStiffness1(l / r)*weight;

				Coord df_i = bulk_ij*p;
				Coord df_j = -bulk_ij*p;

				atomicAdd(&force[pId][0], df_i[0]);
				atomicAdd(&force[pId][1], df_i[1]);
				atomicAdd(&force[pId][2], df_i[2]);
				atomicAdd(&force[j][0], df_j[0]);
				atomicAdd(&force[j][1], df_j[1]);
				atomicAdd(&force[j][2], df_j[2]);

			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition1(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> tmpPos,
		DeviceArray<Coord> accuPos,
		DeviceArray<Real> accuLambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] = (tmpPos[pId] + accuPos[pId]) / (1.0f + accuLambda[pId]);

// 		if (pId % 200 == 0)
// 		{
// 			printf("%f %f %f \n", accuPos[pId][0], accuPos[pId][1], accuPos[pId][2]);
// 		}
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdateVelocity1(
		DeviceArray<Coord> velArr,
		DeviceArray<Coord> prePos,
		DeviceArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] *= 0.8;
	}

	template<typename TDataType>
	HyperelasticForce<TDataType>::HyperelasticForce()
		: ForceModule()
		, m_posID(MechanicalState::position())
		, m_velID(MechanicalState::velocity())
		, m_forceID(MechanicalState::force())
		, m_neighborhoodID(MechanicalState::particle_neighbors())
		, m_initPosID(MechanicalState::init_position())
		, m_posPreID(MechanicalState::pre_position())
		, m_refMatrix(NULL)
		, m_tmpPos(NULL)
		, m_lambdas(NULL)
		, m_accPos(NULL)
		, m_bulkCoef(NULL)
		, m_needUpdate(true)
		, m_horizon(Real(0.0125))
	{
	}

	template<typename TDataType>
	bool HyperelasticForce<TDataType>::applyForce()
	{
		Real dt = getParent()->getDt();

		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for SummationDensity!" << std::endl;
		}

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);
		auto forceFd = mstate->getField<DeviceArrayField<Coord>>(m_forceID);
		auto posPreFd = mstate->getField<DeviceArrayField<Coord>>(m_posPreID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborhoodID);


		uint pDims = cudaGridSize(posFd->getReference()->size(), BLOCK_SIZE);

		int num = posFd->getReference()->size();
		if (NULL == m_refMatrix)
			m_refMatrix = DeviceArrayField<Matrix>::create(num);
		if (NULL == m_tmpPos)
			m_tmpPos = DeviceArrayField<Coord>::create(num);
		if (NULL == m_lambdas)
			m_lambdas = DeviceArrayField<Real>::create(num);
		if (NULL == m_accPos)
			m_accPos = DeviceArrayField<Coord>::create(num);
		if (NULL == m_bulkCoef)
			m_bulkCoef = DeviceArrayField<Real>::create(num);


		auto matArr = m_refMatrix->getReference();
		auto lambda = m_lambdas->getReference();
		auto bulks = m_bulkCoef->getReference();
		auto accPos = m_accPos->getReference();
		auto tmpPos = m_tmpPos->getReference();

		if (isUpdateRequired())
		{
			construct(neighborFd->getValue(), posFd->getValue());
		}

		Function1Pt::copy(*tmpPos, posFd->getValue());
		K_ComputeDeformationTensor <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (*matArr, m_refPos, SmoothKernel<Real>(), m_horizon);

		int total_itoration = 5;
		int itor = 0;
		accPos->reset();
		lambda->reset();
		K_ComputeElasticForce << <pDims, BLOCK_SIZE >> > (
			forceFd->getValue(),
			*accPos,
			*lambda,
			*bulks,
			posFd->getValue(),
			*matArr,
			m_refPos,
			SmoothKernel<Real>(),
			m_horizon);

		K_UpdateVelocity1 << <pDims, BLOCK_SIZE >> > (velFd->getValue(), posPreFd->getValue(), posFd->getValue(), dt);

		return true;
	}

	template <typename Coord, typename NPair>
	__global__ void K_UpdateRestShape1(
		NeighborList<NPair> shape,
		NeighborList<int> nbr,
		DeviceArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		NPair np;
		int nbSize = nbr.getNeighborSize(pId);
		
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(pId, ne);
			np.j = j;
			np.pos = pos[j];
 			if (pId != j)
 			{
 				shape.setElement(pId, ne, np);
			}
			else
			{
				if (ne == 0)
				{
					shape.setElement(pId, ne, np);
				}
				else
				{
					auto ele = shape.getElement(pId, 0);
					shape.setElement(pId, 0, np);
					shape.setElement(pId, ne, ele);
				}
			}
		}
	}

	template<typename TDataType>
	void HyperelasticForce<TDataType>::construct(NeighborList<int>& nbr, DeviceArray<Coord>& pos)
	{
		m_refPos.resize(nbr.size());
		if (nbr.isLimited())
		{
			m_refPos.setNeighborLimit(nbr.getNeighborLimit());
		}

		Function1Pt::copy(m_refPos.getIndex(), nbr.getIndex());

		uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);

		K_UpdateRestShape1 << <pDims, BLOCK_SIZE >> > (m_refPos, nbr, pos);

		m_needUpdate = false;
	}
}