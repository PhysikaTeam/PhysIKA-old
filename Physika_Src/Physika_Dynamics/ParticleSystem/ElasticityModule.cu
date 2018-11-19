#include <cuda_runtime.h>
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Algorithm/MatrixFunc.h"
#include "ElasticityModule.h"

namespace Physika
{
// 	struct EM_STATE
// 	{
// 		float mass;
// 		float smoothingLength;
// 		float samplingDistance;
// 		float restDensity;
// //		SmoothKernel<float> kernSmooth;
// 	};
// 
// 	__constant__ EM_STATE const_em_state;


	__device__ float OneNorm(const glm::mat3 &A)
	{
		const float sum1 = fabs(A[0][0]) + fabs(A[1][0]) + fabs(A[2][0]);
		const float sum2 = fabs(A[0][1]) + fabs(A[1][1]) + fabs(A[2][1]);
		const float sum3 = fabs(A[0][2]) + fabs(A[1][2]) + fabs(A[2][2]);
		float maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}


	/** Return the inf norm of the matrix.
	*/
	__device__ float InfNorm(const glm::mat3 &A)
	{
		const float sum1 = fabs(A[0][0]) + fabs(A[1][0]) + fabs(A[2][0]);
		const float sum2 = fabs(A[0][1]) + fabs(A[1][1]) + fabs(A[2][1]);
		const float sum3 = fabs(A[0][2]) + fabs(A[1][2]) + fabs(A[2][2]);
		float maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}

	__device__ void PolarDecompositionStable(
		const glm::mat3 &M,
		const float tolerance, 
		glm::mat3 &R)
	{
		glm::mat3 Mt = glm::transpose(M);
		float Mone = OneNorm(M);
		float Minf = InfNorm(M);
		float Eone;
		glm::mat3 MadjTt, Et;
		do
		{
			MadjTt[0] = glm::cross(Mt[1], Mt[2]);
			MadjTt[1] = glm::cross(Mt[2], Mt[0]);
			MadjTt[2] = glm::cross(Mt[0], Mt[1]);

			float det = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];

			if (fabs(det) < 1.0e-12)
			{
				glm::vec3 len;
				unsigned int index = 0xffffffff;
				for (unsigned int i = 0; i < 3; i++)
				{
					len[i] = MadjTt[i].length();
					if (len[i] > 1.0e-12)
					{
						// index of valid cross product
						// => is also the index of the vector in Mt that must be exchanged
						index = i;
						break;
					}
				}
				if (index == 0xffffffff)
				{
					R = glm::mat3();
					return;
				}
				else
				{
					Mt[index] = glm::cross(Mt[(index + 1) % 3], Mt[(index + 2) % 3]);
					MadjTt[(index + 1) % 3] = glm::cross(Mt[(index + 2) % 3], Mt[(index) % 3]);;
					MadjTt[(index + 2) % 3] = glm::cross(Mt[(index) % 3], Mt[(index + 1) % 3]);
					glm::mat3 M2 = glm::transpose(Mt);
					Mone = OneNorm(M2);
					Minf = InfNorm(M2);
					det = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];
				}
			}

			const float MadjTone = OneNorm(MadjTt);
			const float MadjTinf = InfNorm(MadjTt);

			const float gamma = sqrt(sqrt((MadjTone*MadjTinf) / (Mone*Minf)) / fabs(det));

			const float g1 = gamma*0.5f;
			const float g2 = 0.5f / (gamma*det);

			for (unsigned char i = 0; i < 3; i++)
			{
				for (unsigned char j = 0; j < 3; j++)
				{
					Et[i][j] = Mt[i][j];
					Mt[i][j] = g1*Mt[i][j] + g2*MadjTt[i][j];
					Et[i][j] -= Mt[i][j];
				}
			}

			Eone = OneNorm(Et);

			Mone = OneNorm(Mt);
			Minf = InfNorm(Mt);
		} while (Eone > Mone * tolerance);

		// Q = Mt^T 
		R = glm::transpose(Mt);
	}

	template <typename Real, typename Coord, typename Matrix, typename RestShape>
	__global__ void EM_PrecomputeShape(
		DeviceArray<Matrix> matArr, 
		DeviceArray<RestShape> restShapes,
		SmoothKernel<Real> kernSmooth,
		Real smoothingLength,
		Real samplingDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= matArr.size()) return;

		Coord rest_i = restShapes[pId].pos[restShapes[pId].idx];
		int size_i = restShapes[pId].size;

		Real total_weight = 0.0f;
		Matrix mat_i = Matrix(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			int j = restShapes[pId].ids[ne];
			Real r = restShapes[pId].distance[ne];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);
				//Vector3f q = (shape_i.pos[ne] - rest_i)*(1.0f/r)*weight;
				Coord q = (restShapes[pId].pos[ne] - rest_i)*sqrt(weight);

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
		//PolarDecompositionNew<Real, Coord, Matrix>(mat_i, R, U, D);
		polarDecomposition(mat_i, R, U, D);

		Real threshold = 0.0001f*samplingDistance;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		mat_i = U.transpose()*D*U*R.transpose();

		matArr[pId] = mat_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename RestShape>
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
	}

	__device__ float EM_GetStiffness(int r)
	{
		return 10.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename RestShape>
	__global__ void EM_EnforceElasticity(
		DeviceArray<Coord> accuPos,
		DeviceArray<Real> accuLamdas,
		DeviceArray<Real> bulkCoefs,
		DeviceArray<Coord> posArr,
		DeviceArray<Matrix> matArr, 
		DeviceArray<RestShape> restShapes,
		SmoothKernel<Real> kernSmooth,
		Real smoothingLength,
		Real samplingDistance)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord rest_i = restShapes[pId].pos[restShapes[pId].idx];
		int size_i = restShapes[pId].size;

		//			cout << i << " " << rids[shape_i.ids[shape_i.idx]] << endl;
		Real total_weight = 0.0f;
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			int j = restShapes[pId].ids[ne];
			Real r = restShapes[pId].distance[ne];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord p = (posArr[j] - posArr[pId]);
				Coord q = (restShapes[pId].pos[ne] - rest_i)*weight;

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

//		if ((deform_i.determinant()) < 0.01f)
// 		{
// 			deform_i = Matrix::identityMatrix();
// 		}
	
		Matrix mat_i = deform_i;
		for (int ne = 0; ne < size_i; ne++)
		{
			int j = restShapes[pId].ids[ne];
			Real r = restShapes[pId].distance[ne];

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord q = (rest_i - restShapes[pId].pos[ne])*(1.0f / r);
				//Coord p = Vec2Float(Float2Vec(q)*mat_i);
				Coord p = mat_i*q;
				//p = normalize(p);

				p.normalize();

				Coord dir_ij = 1.0f*r*p;
				// 					Vector3f q = (rest_i - shape_i.pos[ne]);//*(1.0f/r);
				// 					Vector3f p = mat_i*q;
				// 					//p.Normalize();
				// 					Vector3f dir_ij = 1.0f*p;
				Coord new_pos_i = dir_ij + posArr[j];
				Coord new_pos_j = -dir_ij + posArr[pId];
				Coord dir_i = 1.0f*(new_pos_i - posArr[pId]);//*(samplingDistance/r)*(samplingDistance/r);
				Coord dir_j = 1.0f*(new_pos_j - posArr[j]);//*(samplingDistance/r)*(samplingDistance/r);

				Real l_i = dir_i.norm();

				Real ratio = weight / total_weight;
				Real cc = (samplingDistance / r);
				if (r < 0.8*samplingDistance)
				{
					cc = 1.0 / 0.8f;
				}

				Real bulk_ij = 1.0f*EM_GetStiffness(l_i/r)*ratio*cc*cc;
				Coord vec_ij = bulk_ij*dir_i;

				atomicAdd(&accuLamdas[pId], bulk_ij);
				atomicAdd(&accuLamdas[j], bulk_ij);

				Coord dP_i = bulk_ij*new_pos_i;
				Coord dP_j = bulk_ij*new_pos_j;

				atomicAdd(&accuPos[pId][0], dP_i[0]);
				atomicAdd(&accuPos[pId][1], dP_i[1]);
				atomicAdd(&accuPos[pId][2], dP_i[2]);
				atomicAdd(&accuPos[j][0], dP_j[0]);
				atomicAdd(&accuPos[j][1], dP_j[1]);
				atomicAdd(&accuPos[j][2], dP_j[2]);

			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void EM_UpdatePosition(
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> tmpPos,
		DeviceArray<Coord> accuPos,
		DeviceArray<Real> accuLambda,
		DeviceArray<int> bFixed, 
		DeviceArray<Coord> fixedPts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		if (bFixed[pId] > 0)
		{
			posArr[pId] = fixedPts[pId];
		}
		else
		{
			posArr[pId] = (tmpPos[pId] + accuPos[pId]) / (1.0f + accuLambda[pId]);
		}

// 		if (pId % 200 == 0)
// 		{
// 			printf("%f %f %f \n", accuPos[pId][0], accuPos[pId][1], accuPos[pId][2]);
// 		}
	}

	template <typename Real, typename Coord>
	__global__ void EM_UpdateVelocity(
		DeviceArray<Coord> velArr,
		DeviceArray<Coord> prePos,
		DeviceArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] = (curPos[pId] - prePos[pId]) / dt;
	}

	template<typename TDataType>
	ElasticityModule<TDataType>::ElasticityModule()
		: ConstraintModule()
		, m_refMatrix(NULL)
		, m_tmpPos(NULL)
		, m_lambdas(NULL)
		, m_accPos(NULL)
		, m_bulkCoef(NULL)
	{
	}

	template<typename TDataType>
	bool ElasticityModule<TDataType>::execute()
	{
		Real dt = getParent()->getDt();

		DeviceArray<Coord>* posArr = m_position.getField().getDataPtr();
		DeviceArray<Coord>* velArr = m_velocity.getField().getDataPtr();
		DeviceArray<Coord>* prePosArr = m_prePosition.getField().getDataPtr();

		uint pDims = cudaGridSize(posArr->size(), BLOCK_SIZE);
		
		int num = posArr->size();
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

		DeviceArray<int>* states = m_state.getField().getDataPtr();
		DeviceArray<Coord>* initPos = m_initPosition.getField().getDataPtr();

		DeviceArray<RestShape>* restShapeArr = m_restShape.getField().getDataPtr();

		DeviceArray<Matrix>* matArr = m_refMatrix->getDataPtr();

		DeviceArray<Real>* lambda = m_lambdas->getDataPtr();
		DeviceArray<Real>* bulks = m_bulkCoef->getDataPtr();
		DeviceArray<Coord>* accPos = m_accPos->getDataPtr();
		DeviceArray<Coord>* tmpPos = m_tmpPos->getDataPtr();

		Real smoothingLength = m_radius.getField().getValue();
		Real samplingDistance = m_samplingDistance.getField().getValue();
		Function1Pt::Copy(*tmpPos, *posArr);
		EM_PrecomputeShape <Real, Coord, Matrix, RestShape> << <pDims, BLOCK_SIZE >> > (*matArr, *restShapeArr, SmoothKernel<Real>(), smoothingLength, samplingDistance);

		int total_itoration = 5;
		int itor = 0;
		while (itor < total_itoration)
		{
			accPos->reset();
			lambda->reset();
			EM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
				*accPos, 
				*lambda, 
				*bulks, 
				*posArr, 
				*matArr, 
				*restShapeArr, 
				SmoothKernel<Real>(), 
				smoothingLength, 
				samplingDistance);
			EM_UpdatePosition << <pDims, BLOCK_SIZE >> > (*posArr, *tmpPos, *accPos, *lambda, *states, *initPos);
			itor++;
		}

//		EM_RotateRestShape << <pDims, BLOCK_SIZE >> > (*posArr, *matArr, *restShapeArr);

		EM_UpdateVelocity << <pDims, BLOCK_SIZE >> > (*velArr, *prePosArr, *posArr, dt);

		return true;
	}
}