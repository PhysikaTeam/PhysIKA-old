#include <cuda_runtime.h>
#include "ElastoplasticityModule.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Core/Utility.h"
#include "Kernel.h"
//#include "svd3_cuda2.h"

namespace Physika
{
	template<typename TDataType>
	ElastoplasticityModule<TDataType>::ElastoplasticityModule()
		: ElasticityModule<TDataType>()
	{
		this->attachField(&m_c, "c", "cohesion!", false);
		this->attachField(&m_phi, "phi", "friction angle!", false);

		m_c.setValue(0.001);
		m_phi.setValue(60.0 / 180.0);
	}

	__device__ Real Hardening(Real rho)
	{
		Real hardening = 1.0f;

		return 1.0f;

		if (rho >= 1000)
		{
			Real ratio = rho / 1000;
			return pow((float)M_E, hardening*(ratio - 1.0f));
		}
		else
			return 1.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ComputeInvariants(
		DeviceArray<bool> bYield,
		DeviceArray<Real> yield_I1,
		DeviceArray<Real> yield_J2,
		DeviceArray<Real> arrI1,
		DeviceArray<Coord> position,
		DeviceArray<Real> density,
		DeviceArray<Real> bulk_stiffiness,
		NeighborList<NPair> restShape,
		Real horizon,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		CorrectedKernel<Real> kernSmooth;

		Real weaking = 1.0f;// Softening(rhoArr[i]);

		Real s_A = weaking*A;

		Coord rest_pos_i = restShape.getElement(i, 0).pos;
		Coord cur_pos_i = position[i];
		
		Real I1_i = 0.0f;
		Real J2_i = 0.0f;
		//compute the first and second invariants of the deformation state, i.e., I1 and J2
		int size_i = restShape.getNeighborSize(i);
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Coord p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;

				I1_i += weight*ratio_ij;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			I1_i /= total_weight;
		}
		else
		{
			I1_i = 1.0f;
		}

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Vector3f p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;
				J2_i = (ratio_ij - I1_i)*(ratio_ij - I1_i)*weight;
			}
		}
		if (total_weight > EPSILON)
		{
			J2_i /= total_weight;
			J2_i = sqrt(J2_i);
		}
		else
		{
			J2_i = 0.0f;
		}

		Real D1 = 1 - I1_i;		//positive for compression and negative for stretching

		Real yield_I1_i = 0.0f;
		Real yield_J2_i = 0.0f;

		Real s_J2 = J2_i*mu*bulk_stiffiness[i];
		Real s_D1 = D1*lambda*bulk_stiffiness[i];

		//Drucker-Prager yield criterion
		if (s_J2 <= s_A + B*s_D1)
		{
			//bulk_stiffiness[i] = 10.0f;
			//invDeform[i] = Matrix::identityMatrix();
			yield_I1[i] = Real(0);
			yield_J2[i] = Real(0);

			bYield[i] = false;
		}
		else
		{
			//bulk_stiffiness[i] = 0.0f;
			if (s_A + B*s_D1 > 0.0f)
			{
				yield_I1_i = 0.0f;

				yield_J2_i = (s_J2 - (s_A + B*s_D1)) / s_J2;
			}
			else
			{
				yield_I1_i = 1.0f;
				if (s_A + B*s_D1 < -EPSILON)
				{
					yield_I1_i = (s_A + B*s_D1) / (B*s_D1);
				}

				yield_J2_i = 1.0f;
			}

			yield_I1[i] = yield_I1_i;
			yield_J2[i] = yield_J2_i;
		}

// 		invDeform[i] = Matrix::identityMatrix();
// 		bYield[i] = true;

//		int size_i = restShape.getNeighborSize(i);
/*		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_pos_i - rest_pos_j).norm();

			Coord p = (position[j] - pos_i);
			Coord q = (rest_pos_j - rest_pos_i);

			Coord new_q = q*I1_i;
			Coord D_iso = new_q - q;
			Coord D_dev = p - new_q;

			NPair new_np_j;

			Coord new_rest_pos_j = rest_pos_j + yield_I1_i * D_iso + yield_J2_i * D_dev;

			new_np_j.pos = new_rest_pos_j;
			new_np_j.index = j;
			restShape.setElement(i, ne, new_np_j);
		}*/

		
		arrI1[i] = I1_i;
// 		if (yield_I1_i > EPSILON || yield_J2_i > EPSILON)
// 		{
// 			printf("%d: %f %f; I1: %f J2: %f \n", i, yield_I1_i, yield_J2_i, I1_i, J2_i);
// 		}
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ApplyPlasticity(
		DeviceArray<Real> yield_I1,
		DeviceArray<Real> yield_J2,
		DeviceArray<Real> arrI1,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShape)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		Coord rest_pos_i = restShape.getElement(i, 0).pos;
		Coord pos_i = position[i];

		Real yield_I1_i = yield_I1[i];
		Real yield_J2_i = yield_J2[i];
		Real I1_i = arrI1[i];

		//add permanent deformation
		int size_i = restShape.getNeighborSize(i);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;

			Real yield_I1_j = yield_I1[j];
			Real yield_J2_j = yield_J2[j];
			Real I1_j = arrI1[j];

			Real r = (rest_pos_i - rest_pos_j).norm();

			Coord p = (position[j] - pos_i);
			Coord q = (rest_pos_j - rest_pos_i);

			//Coord new_q = q*I1_i;
			Coord new_q = q*(I1_i + I1_j) / 2;
			Coord D_iso = new_q - q;

			Coord dir_q = q;
			dir_q = dir_q.norm() > EPSILON ? dir_q.normalize() : Coord(0);

			Coord D_dev = p.norm()*dir_q - new_q;
			//Coord D_dev = p - new_q;

			NPair new_np_j;

			//Coord new_rest_pos_j = rest_pos_j + yield_I1_i * D_iso + yield_J2_i * D_dev;
			Coord new_rest_pos_j = rest_pos_j + (yield_I1_i + yield_I1_j) / 2 * D_iso + (yield_J2_i + yield_J2_j) / 2 *D_dev;

// 			if ((new_rest_pos_j-rest_pos_j).norm() > 0.002)
// 			{
// 				printf("Error---------------------------------- \n; yield_I1: %f %f %f %f; yield_J2 %f %f %f %f; %f; Norm: %f; new_q: %f %f %f; I1_i: %f \n", yield_I1_i, D_iso[0], D_iso[1], D_iso[2], yield_J2_i, D_dev[0], D_dev[1], D_dev[2], I1_i, p.norm(), new_q[0], new_q[1], new_q[2], I1_i);
// 			}

			new_np_j.pos = new_rest_pos_j;
			new_np_j.index = j;
			restShape.setElement(i, ne, new_np_j);
		}

	}


//	int iter = 0;
	template<typename TDataType>
	bool ElastoplasticityModule<TDataType>::constrain()
	{
		solveElasticity();
		applyPlasticity();

		reconstructRestShape();

		return true;
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::applyPlasticity()
	{
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);
// 
		rotateRestShape();

		Real A = computeA();
		Real B = computeB();

		PM_ComputeInvariants<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_bYield,
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			m_position.getValue(),
			this->getDensity(),
			m_bulkCoefs,
			m_restShape.getValue(),
			m_horizon.getValue(),
			A,
			B,
			m_mu.getValue(),
			m_lambda.getValue());
		cuSynchronize();
// 
		PM_ApplyPlasticity<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			m_position.getValue(),
			m_restShape.getValue());
		cuSynchronize();
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ReconstructRestShape(
		NeighborList<NPair> new_rest_shape,
		DeviceArray<bool> bYield,
		DeviceArray<Coord> position,
		DeviceArray<Real> I1,
		DeviceArray<Real> I1_yield,
		DeviceArray<Real> J2_yield,
		DeviceArray<Matrix> invF,
		NeighborList<int> neighborhood,
		NeighborList<NPair> restShape,
		Real horizon)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= new_rest_shape.size()) return;

		// update neighbors
		if (!bYield[i])
		{
			Coord rest_pos_i = restShape.getElement(i, 0).pos;

			int new_size = restShape.getNeighborSize(i);
			for (int ne = 0; ne < new_size; ne++)
			{
				NPair pair = restShape.getElement(i, ne);
				new_rest_shape.setElement(i, ne, pair);
			}
		}
		else
		{
			int nbSize = neighborhood.getNeighborSize(i);
			Coord pos_i = position[i];

			Matrix invF_i = invF[i];

			NPair np;
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = neighborhood.getElement(i, ne);
				Matrix invF_j = invF[j];

				np.index = j;
				np.pos = pos_i + 0.5*(invF_i+invF_j)*(position[j] - pos_i);
				if (i != j)
				{
					new_rest_shape.setElement(i, ne, np);
				}
				else
				{
					if (ne == 0)
					{
						new_rest_shape.setElement(i, ne, np);
					}
					else
					{
						auto ele = new_rest_shape.getElement(i, 0);
						new_rest_shape.setElement(i, 0, np);
						new_rest_shape.setElement(i, ne, ele);
					}
				}
			}
		}

		bYield[i] = false;
	}

	template <typename NPair>
	__global__ void PM_ReconfigureRestShape(
		DeviceArray<int> nbSize,
		DeviceArray<bool> bYield,
		NeighborList<int> neighborhood,
		NeighborList<NPair> restShape)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= nbSize.size()) return;

		if (bYield[i])
		{
			nbSize[i] = neighborhood.getNeighborSize(i);
		}
		else
		{
			nbSize[i] = restShape.getNeighborSize(i);
		}
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ComputeInverseDeformation(
		DeviceArray<Matrix> invF,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShape,
		Real horizon)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= invF.size()) return;

		CorrectedKernel<Real> kernSmooth;

		//reconstruct the rest shape as the yielding condition is violated.
		Real total_weight = 0.0f;
		Matrix curM(0);
		Matrix refM(0);
		Coord rest_pos_i = restShape.getElement(i, 0).pos;
		int size_i = restShape.getNeighborSize(i);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			int j = np_j.index;
			Coord rest_j = np_j.pos;
			Real r = (rest_j - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Coord p = (position[j] - position[i]) / horizon;
				Coord q = (rest_j - rest_pos_i) / horizon;

				curM(0, 0) += p[0] * p[0] * weight; curM(0, 1) += p[0] * p[1] * weight; curM(0, 2) += p[0] * p[2] * weight;
				curM(1, 0) += p[1] * p[0] * weight; curM(1, 1) += p[1] * p[1] * weight; curM(1, 2) += p[1] * p[2] * weight;
				curM(2, 0) += p[2] * p[0] * weight; curM(2, 1) += p[2] * p[1] * weight; curM(2, 2) += p[2] * p[2] * weight;

				refM(0, 0) += q[0] * p[0] * weight; refM(0, 1) += q[0] * p[1] * weight; refM(0, 2) += q[0] * p[2] * weight;
				refM(1, 0) += q[1] * p[0] * weight; refM(1, 1) += q[1] * p[1] * weight; refM(1, 2) += q[1] * p[2] * weight;
				refM(2, 0) += q[2] * p[0] * weight; refM(2, 1) += q[2] * p[1] * weight; refM(2, 2) += q[2] * p[2] * weight;

				total_weight += weight;
			}
		}


		if (total_weight < EPSILON)
		{
			total_weight = Real(1);
		}
		refM *= (1.0f / total_weight);
		curM *= (1.0f / total_weight);

		Real threshold = Real(0.00001);
		Matrix curR, curU, curD, curV;

		polarDecomposition(curM, curR, curU, curD, curV);

		curD(0, 0) = curD(0, 0) > threshold ? 1.0 / curD(0, 0) : 1.0 / threshold;
		curD(1, 1) = curD(1, 1) > threshold ? 1.0 / curD(1, 1) : 1.0 / threshold;
		curD(2, 2) = curD(2, 2) > threshold ? 1.0 / curD(2, 2) : 1.0 / threshold;
		refM *= curV*curD*curU.transpose();

// 		if (abs(refM.determinant() - 1) > 0.05f)
// 		{
// 			refM = Matrix::identityMatrix();
// 		}

		if (refM.determinant() < EPSILON)
		{
			refM = Matrix::identityMatrix();
		}

// 		if (i == 20)
// 		{
// 			printf("PM_ComputeInverseDeformation***************************** \n\n");
// 
// 			printf("Invserse F: \n %f %f %f \n %f %f %f \n %f %f %f \n	Determinant: %f \n\n",
// 				refM(0, 0), refM(0, 1), refM(0, 2),
// 				refM(1, 0), refM(1, 1), refM(1, 2),
// 				refM(2, 0), refM(2, 1), refM(2, 2),
// 				refM.determinant());
// 		}

		invF[i] = refM;
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::reconstructRestShape()
	{
		//constructRestShape(m_neighborhood.getValue(), m_position.getValue());

		uint pDims = cudaGridSize(m_position.getElementCount(), BLOCK_SIZE);

		NeighborList<NPair> newNeighborList;
		newNeighborList.resize(m_position.getElementCount());
		DeviceArray<int>& index = newNeighborList.getIndex();
		DeviceArray<NPair>& elements = newNeighborList.getElements();

		PM_ReconfigureRestShape << <pDims, BLOCK_SIZE >> > (
			index,
			m_bYield,
			m_neighborhood.getValue(),
			m_restShape.getValue());

		int total_num = thrust::reduce(thrust::device, index.getDataPtr(), index.getDataPtr() + index.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, index.getDataPtr(), index.getDataPtr() + index.size(), index.getDataPtr());
		elements.resize(total_num);

		PM_ComputeInverseDeformation << <pDims, BLOCK_SIZE >> > (
			m_invF,
			m_position.getValue(),
			m_restShape.getValue(),
			m_horizon.getValue());

		PM_ReconstructRestShape<< <pDims, BLOCK_SIZE >> > (
			newNeighborList,
			m_bYield,
			m_position.getValue(),
			m_I1,
			m_yiled_I1,
			m_yield_J2,
			m_invF,
			m_neighborhood.getValue(),
			m_restShape.getValue(),
			m_horizon.getValue());

		m_restShape.getValue().copyFrom(newNeighborList);

		newNeighborList.release();
		cuSynchronize();
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_RotateRestShape(
		DeviceArray<Coord> position,
		DeviceArray<bool> bYield,
		NeighborList<NPair> restShapes,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		SmoothKernel<Real> kernSmooth;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		//			cout << i << " " << rids[shape_i.ids[shape_i.idx]] << endl;
		Real total_weight = 0.0f;
		Matrix mat_i(0);
		Matrix invK_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord p = (position[j] - position[pId]) / smoothingLength;
				//Vector3f q = (shape_i.pos[ne] - rest_i)*(1.0f/r)*weight;
				Coord q = (rest_pos_j - rest_pos_i) / smoothingLength;

				mat_i(0, 0) += p[0] * q[0] * weight; mat_i(0, 1) += p[0] * q[1] * weight; mat_i(0, 2) += p[0] * q[2] * weight;
				mat_i(1, 0) += p[1] * q[0] * weight; mat_i(1, 1) += p[1] * q[1] * weight; mat_i(1, 2) += p[1] * q[2] * weight;
				mat_i(2, 0) += p[2] * q[0] * weight; mat_i(2, 1) += p[2] * q[1] * weight; mat_i(2, 2) += p[2] * q[2] * weight;

				invK_i(0, 0) += q[0] * q[0] * weight; invK_i(0, 1) += q[0] * q[1] * weight; invK_i(0, 2) += q[0] * q[2] * weight;
				invK_i(1, 0) += q[1] * q[0] * weight; invK_i(1, 1) += q[1] * q[1] * weight; invK_i(1, 2) += q[1] * q[2] * weight;
				invK_i(2, 0) += q[2] * q[0] * weight; invK_i(2, 1) += q[2] * q[1] * weight; invK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
			invK_i *= (1.0f / total_weight);
		}

// 		if (pId == 0)
// 		{
// 			printf("RotateRestShape**************************************");
// 
// 			printf("invK: \n %f %f %f \n %f %f %f \n %f %f %f \n\n\n",
// 				invK_i(0, 0), invK_i(0, 1), invK_i(0, 2),
// 				invK_i(1, 0), invK_i(1, 1), invK_i(1, 2),
// 				invK_i(2, 0), invK_i(2, 1), invK_i(2, 2));
// 
// 			printf("mat_i: \n %f %f %f \n %f %f %f \n %f %f %f \n\n\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		Matrix R, U, D, V;
		polarDecomposition(invK_i, R, U, D, V);

// 		svd(invK_i(0, 0), invK_i(0, 1), invK_i(0, 2), invK_i(1, 0), invK_i(1, 1), invK_i(1, 2), invK_i(2, 0), invK_i(2, 1), invK_i(2, 2),
// 			R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2),
// 			D(0, 0), D(1, 1), D(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2));

		Real threshold = 0.0001f*smoothingLength;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		//invK_i = R.transpose()*U*D*U.transpose();
		invK_i = V*D*U.transpose();

		mat_i *= invK_i;

//		polarDecomposition(mat_i, R, Real(EPSILON));

// 		if (pId == 0)
// 		{
// 			printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	Determinant: %f \n\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 				R.determinant());
// 		}
		

		polarDecomposition(mat_i, R, U, D, V);

// 		if (pId == 20)
// 		{
// 			Matrix rMat_i = U*D*V.transpose();
// 			printf("rMat: \n %f %f %f \n %f %f %f \n %f %f %f \n	Determinant: %f \n\n",
// 				rMat_i(0, 0), rMat_i(0, 1), rMat_i(0, 2),
// 				rMat_i(1, 0), rMat_i(1, 1), rMat_i(1, 2),
// 				rMat_i(2, 0), rMat_i(2, 1), rMat_i(2, 2),
// 				R.determinant());
// 
// 			printf("R: \n %f %f %f \n %f %f %f \n %f %f %f \n	Determinant: %f \n\n",
// 				R(0, 0), R(0, 1), R(0, 2),
// 				R(1, 0), R(1, 1), R(1, 2),
// 				R(2, 0), R(2, 1), R(2, 2),
// 				R.determinant());
// 
// 			Matrix rR;
// 			polarDecomposition(rMat_i, rR, U, D);
// 
// 			printf("pre R: \n %f %f %f \n %f %f %f \n %f %f %f \n	Determinant: %f \n\n",
// 				rR(0, 0), rR(0, 1), rR(0, 2),
// 				rR(1, 0), rR(1, 1), rR(1, 2),
// 				rR(2, 0), rR(2, 1), rR(2, 2),
// 				rR.determinant());
// 		}

// 		svd(mat_i(0, 0), mat_i(0, 1), mat_i(0, 2), mat_i(1, 0), mat_i(1, 1), mat_i(1, 2), mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 			R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2),
// 			D(0, 0), D(1, 1), D(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2));

// 		D(0, 0) = Real(1);
// 		D(1, 1) = Real(1);
// 		D(2, 2) = (R*U.transpose()).determinant();

//		R = U*D*R.transpose();

// 		printf("\n \n Rotation: \n %f %f %f \n %f %f %f \n %f %f %f \n", 
// 			R(0, 0), R(0, 1), R(0, 2),
// 			R(1, 0), R(1, 1), R(1, 2),
// 			R(2, 0), R(2, 1), R(2, 2));

// 		if (R.determinant() < 0.9)
// 		{
// 			bYield[pId] = true;
// 		}
//		bYield[pId] = true;
		// 		mat_i(0, 0) = 0.940038; mat_i(0, 1) = 0; mat_i(0, 2) = 0;
		// 		mat_i(1, 0) = 0.001991; mat_i(1, 1) = 0; mat_i(1, 2) = 0;
		// 		mat_i(2, 0) = 0; mat_i(2, 1) = 0; mat_i(2, 2) = 0;

		// 		mat_i(0, 0) = 0.115572; mat_i(0, 1) = 0.022244; mat_i(0, 2) = 0.188606;
		// 		mat_i(1, 0) = -0.062891; mat_i(1, 1) = 0.012105; mat_i(1, 2) = -0.102634;
		// 		mat_i(2, 0) = 0.120823; mat_i(2, 1) = -0.023255; mat_i(2, 2) = 0.197176;

		//		polarDecomposition(mat_i, R, U, D);

		// 		if (pId == 10)
		// 		{
		// 			printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	R: \n %f %f %f \n %f %f %f \n %f %f %f \n U :\n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
		// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
		// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
		// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
		// 				R(0, 0), R(0, 1), R(0, 2),
		// 				R(1, 0), R(1, 1), R(1, 2),
		// 				R(2, 0), R(2, 1), R(2, 2),
		// 				U(0, 0), U(0, 1), U(0, 2),
		// 				U(1, 0), U(1, 1), U(1, 2),
		// 				U(2, 0), U(2, 1), U(2, 2),
		// 				R.determinant());
		// 		}



		//		if (R.determinant() < 0.9)
		//		{

		// 
		// 			printf("determinant: %f \n", R.determinant());

		//			R = Matrix::identityMatrix();
		//		}

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;

			Coord new_rest_pos_j = rest_pos_i + R*(rest_pos_j - rest_pos_i);
			np_j.pos = new_rest_pos_j;
			restShapes.setElement(pId, ne, np_j);
		}
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::rotateRestShape()
	{
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		EM_RotateRestShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_position.getValue(),
			m_bYield,
			m_restShape.getValue(),
			m_horizon.getValue());
		cuSynchronize();
	}

	template<typename TDataType>
	bool ElastoplasticityModule<TDataType>::initializeImpl()
	{
		m_invF.resize(m_position.getElementCount());
		m_yiled_I1.resize(m_position.getElementCount());
		m_yield_J2.resize(m_position.getElementCount());
		m_I1.resize(m_position.getElementCount());
		m_bYield.resize(m_position.getElementCount());

		m_bYield.reset();
		return ElasticityModule<TDataType>::initializeImpl();
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::setCohesion(Real c)
	{
		m_c.setValue(c);
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::setFrictionAngle(Real phi)
	{
		m_phi.setValue(phi/180);
	}


	template <typename Real>
	__global__ void PM_ComputeStiffness(
		DeviceArray<Real> stiffiness,
		DeviceArray<Real> density)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= stiffiness.size()) return;

		if (density[i] < 1000)
		{
			stiffiness[i] = 0;
		}
		else
		{
			stiffiness[i] = 1;
		}
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::computeStiffness()
	{
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		PM_ComputeStiffness<< <pDims, BLOCK_SIZE >> > (
			m_bulkCoefs,
			this->getDensity());
		cuSynchronize();
	}

}