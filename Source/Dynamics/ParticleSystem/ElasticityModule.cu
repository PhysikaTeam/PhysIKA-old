#include <cuda_runtime.h>
#include "ElasticityModule.h"
#include "Framework/Framework/Node.h"
#include "Core/Algorithm/MatrixFunc.h"
#include "Core/Utility.h"
#include "Kernel.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ElasticityModule, TDataType)

	template<typename Real>
	__device__ Real D_Weight(Real r, Real h)
	{
		SmoothKernel<Real> kernSmooth;
		Real q = r / h;
		return q*q*kernSmooth.Weight(r, h);
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_PrecomputeShape(
		DeviceArray<Matrix> invK,
		NeighborList<NPair> restShapes,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= invK.size()) return;

		CorrectedKernel<float> g_weightKernel;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = 0.0f;
		Matrix mat_i = Matrix(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			Real r = (rest_i - rest_j).norm();

			if (r > EPSILON)
			{
				Real weight = g_weightKernel.Weight(r, smoothingLength);
				Coord q = (rest_j - rest_i) / smoothingLength*sqrt(weight);

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

		Matrix R(0), U(0), D(0), V(0);

// 		if (pId == 0)
// 		{
// 			printf("EM_PrecomputeShape**************************************");
// 
// 			printf("K: \n %f %f %f \n %f %f %f \n %f %f %f \n\n\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		polarDecomposition(mat_i, R, U, D, V);

		Real threshold = 0.0001f*smoothingLength;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		mat_i = V*D*U.transpose();

// 		polarDecomposition(mat_i, R, U, D);
// 
// 		Real threshold = 0.0001f*smoothingLength;
// 		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
// 		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
// 		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
// 
// 		mat_i = R.transpose()*U*D*U.transpose();

// 		printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	R: \n %f %f %f \n %f %f %f \n %f %f %f \n D: \n %f %f %f \n %f %f %f \n %f %f %f \n U :\n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
// 			mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 			mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 			mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 			R(0, 0), R(0, 1), R(0, 2),
// 			R(1, 0), R(1, 1), R(1, 2),
// 			R(2, 0), R(2, 1), R(2, 2),
// 			D(0, 0), D(0, 1), D(0, 2),
// 			D(1, 0), D(1, 1), D(1, 2),
// 			D(2, 0), D(2, 1), D(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2),
// 			U(1, 0), U(1, 1), U(1, 2),
// 			U(2, 0), U(2, 1), U(2, 2),
// 			R.determinant());
// 		printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	U :\n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
// 			mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 			mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 			mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2),
// 			U(1, 0), U(1, 1), U(1, 2),
// 			U(2, 0), U(2, 1), U(2, 2),
// 			R.determinant());

		invK[pId] = mat_i;
	}

	__device__ float EM_GetStiffness(int r)
	{
		return 10.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_EnforceElasticity(
		DeviceArray<Coord> delta_position,
		DeviceArray<Real> weights,
		DeviceArray<Real> bulkCoefs,
		DeviceArray<Matrix> invK,
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mu,
		Real lambda)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Coord cur_pos_i = position[pId];

		Coord accPos = Coord(0);
		Real accA = Real(0);
		Real bulk_i = bulkCoefs[pId];


		Real total_weight = 0.0f;
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = g_weightKernel.Weight(r, horizon);

				Coord p = (position[j] - position[pId]) / horizon;
				Coord q = (rest_j - rest_i) / horizon*weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}


		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
			deform_i = deform_i * invK[pId];
		}
		else
		{
			total_weight = 1.0f;
		}

		//Check whether the reference shape is inverted, if yes, simply set K^{-1} to be an identity matrix
		//Note other solutions are possible.
		if ((deform_i.determinant()) < -0.001f)
		{
			deform_i = Matrix::identityMatrix();
		}


		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Coord cur_pos_j = position[j];
			Real r = (rest_j - rest_i).norm();

			if (r > 0.01f*horizon)
			{
				Real weight = g_weightKernel.WeightRR(r, horizon);

				Coord rest_dir_ij = deform_i*(rest_i - rest_j);
				Coord cur_dir_ij = cur_pos_i - cur_pos_j;

				cur_dir_ij = cur_dir_ij.norm() > EPSILON ? cur_dir_ij.normalize() : Coord(0);
				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				Real mu_ij = mu*bulk_i* g_weightKernel.WeightRR(r, horizon);
				Coord mu_pos_ij = position[j] + r*rest_dir_ij;
				Coord mu_pos_ji = position[pId] - r*rest_dir_ij;

				Real lambda_ij = lambda*bulk_i*g_weightKernel.WeightRR(r, horizon);
				Coord lambda_pos_ij = position[j] + r*cur_dir_ij;
				Coord lambda_pos_ji = position[pId] - r*cur_dir_ij;

				Coord delta_pos_ij = mu_ij*mu_pos_ij + lambda_ij*lambda_pos_ij;
				Real delta_weight_ij = mu_ij + lambda_ij;

				Coord delta_pos_ji = mu_ij*mu_pos_ji + lambda_ij*lambda_pos_ji;

				accA += delta_weight_ij;
				accPos += delta_pos_ij;


				atomicAdd(&weights[j], delta_weight_ij);
				atomicAdd(&delta_position[j][0], delta_pos_ji[0]);
				atomicAdd(&delta_position[j][1], delta_pos_ji[1]);
				atomicAdd(&delta_position[j][2], delta_pos_ji[2]);
			}
		}

		atomicAdd(&weights[pId], accA);
		atomicAdd(&delta_position[pId][0], accPos[0]);
		atomicAdd(&delta_position[pId][1], accPos[1]);
		atomicAdd(&delta_position[pId][2], accPos[2]);
	}


	template <typename Real, typename Coord, typename NPair>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> position,
		DeviceArray<Coord> delta_position,
		NeighborList<NPair> restShapes,
		Real horizon)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		CorrectedKernel<float> g_weightKernel;
		Coord delta_pos_i = delta_position[pId];

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;

		Coord new_delta_pos_i = Coord(0);
		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_pos_j - rest_pos_i).norm();

			Coord delta_pos_j = delta_position[j];

			new_delta_pos_i += 0.1*(delta_pos_i)*g_weightKernel.Weight(r, horizon);
		}



//		position[pId] += delta_pos_i;
		position[pId] += delta_position[pId];

	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> position,
		DeviceArray<Coord> old_position,
		DeviceArray<Coord> delta_position,
		DeviceArray<Real> delta_weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = (old_position[pId] + delta_position[pId]) / (1.0+delta_weights[pId]);
	}


	template <typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
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
	ElasticityModule<TDataType>::ElasticityModule()
		: ConstraintModule()
	{
//		this->attachField(&m_horizon, "horizon", "Supporting radius!", false);
//		this->attachField(&m_distance, "distance", "The sampling distance!", false);
		this->attachField(&m_mu, "mu", "Material stiffness!", false);
		this->attachField(&m_lambda, "lambda", "Material stiffness!", false);
		this->attachField(&m_iterNum, "Iterations", "Iteration Number", false);

//		this->attachField(&m_position, "position", "Storing the particle positions!", false);
//		this->attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
//		this->attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);

//		this->attachField(&testing, "testing", "For testing", false);
//		this->attachField(&TetOut, "TetOut", "For testing", false);

		this->inHorizon()->setValue(0.0125);
 		m_mu.setValue(0.05);
 		m_lambda.setValue(0.1);
		m_iterNum.setValue(10);
	}


	template<typename TDataType>
	ElasticityModule<TDataType>::~ElasticityModule()
	{
		m_weights.release();
		m_displacement.release();
		m_invK.release();
		m_F.release();
		m_position_old.release();
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::enforceElasticity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_displacement.reset();
		m_weights.reset();

		EM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
			m_displacement,
			m_weights,
			m_bulkCoefs,
			m_invK,
			this->inPosition()->getValue(),
			m_restShape.getValue(),
			this->inHorizon()->getValue(),
			m_mu.getValue(),
			m_lambda.getValue());
		cuSynchronize();

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(),
			m_position_old,
			m_displacement,
			m_weights);
		cuSynchronize();
	}

	template<typename Real>
	__global__ void EM_InitBulkStiffness(DeviceArray<Real> stiffness)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= stiffness.size()) return;

		stiffness[pId] = Real(1);
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::computeMaterialStiffness()
	{
		int num = this->inPosition()->getElementCount();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		EM_InitBulkStiffness << <pDims, BLOCK_SIZE >> > (m_bulkCoefs);
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::computeInverseK()
	{
		int num = m_restShape.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		EM_PrecomputeShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_invK,
			m_restShape.getValue(),
			this->inHorizon()->getValue());
		cuSynchronize();
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::solveElasticity()
	{
		//Save new positions
		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		this->computeInverseK();

		int itor = 0;
		while (itor < m_iterNum.getValue())
		{
			this->enforceElasticity();

			itor++;
		}

		this->updateVelocity();
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real dt = this->getParent()->getDt();

		K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
			this->inVelocity()->getValue(),
			m_position_old,
			this->inPosition()->getValue(),
			dt);
		cuSynchronize();
	}


	template<typename TDataType>
	bool ElasticityModule<TDataType>::constrain()
	{
		this->solveElasticity();

		return true;
	}


	template <typename Coord, typename NPair>
	__global__ void K_UpdateRestShape(
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
			np.index = j;
			np.pos = pos[j];
 			if (pId != j)
 			{
// 				if (pId == 4 && j == 5)
// 				{
// 					np.pos += Coord(0.0001, 0, 0);
// 				}
// 
// 				if (pId == 5 && j == 4)
// 				{
// 					np.pos += Coord(-0.0001, 0, 0);
// 				}

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
	void ElasticityModule<TDataType>::resetRestShape()
	{
		m_restShape.setElementCount(this->inNeighborhood()->getValue().size());
		m_restShape.getValue().getIndex().resize(this->inNeighborhood()->getValue().getIndex().size());

		if (this->inNeighborhood()->getValue().isLimited())
		{
			m_restShape.getValue().setNeighborLimit(this->inNeighborhood()->getValue().getNeighborLimit());
		}
		else
		{
			m_restShape.getValue().getElements().resize(this->inNeighborhood()->getValue().getElements().size());
		}

		Function1Pt::copy(m_restShape.getValue().getIndex(), this->inNeighborhood()->getValue().getIndex());

		uint pDims = cudaGridSize(this->inPosition()->getValue().size(), BLOCK_SIZE);

		K_UpdateRestShape<< <pDims, BLOCK_SIZE >> > (m_restShape.getValue(), this->inNeighborhood()->getValue(), this->inPosition()->getValue());
		cuSynchronize();
	}

	template<typename TDataType>
	bool ElasticityModule<TDataType>::initializeImpl()
	{
		if (this->inHorizon()->isEmpty() || this->inPosition()->isEmpty() || this->inVelocity()->isEmpty() || this->inNeighborhood()->isEmpty())
		{
			std::cout << "Exception: " << std::string("ElasticityModule's fields are not fully initialized!") << "\n";
			return false;
		}

		int num = this->inPosition()->getElementCount();
		
		m_invK.resize(num);
		m_weights.resize(num);
		m_displacement.resize(num);

		m_F.resize(num);
		
		m_position_old.resize(num);
		m_bulkCoefs.resize(num);

		resetRestShape();

		this->computeMaterialStiffness();

		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		return true;
	}

}