#include "ShallowWaterEquationModel.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/Helmholtz.h"
#include "Dynamics/ParticleSystem/Attribute.h"
#include "Core/Utility.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ShallowWaterEquationModel, TDataType)

	template<typename TDataType>
	ShallowWaterEquationModel<TDataType>::ShallowWaterEquationModel()
		: NumericalModel()
		, m_pNum(0)
	{
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&grid_vel_x, "grid_vel_x", "Storing the grid velocities x!", false);
		attachField(&grid_vel_z, "grid_vel_z", "Storing the grid velocities z!", false);

		//attachField(&m_force, "force", "Storing the particle force densities!", false);

		attachField(&solid, "solid", "Storing the solid grid!", false);
		attachField(&normal, "solidnormal", "Storing the solid normal!", false);
		attachField(&isBound, "isBound", "Storing the solid isBound!", false);
		attachField(&h, "h", "Storing the water height!", false);
	}

	template<typename Real, typename Coord>
	__global__ void Init(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> solid,
		DeviceArray<Real> h,
		DeviceArray<int> isBound,
		DeviceArray<Coord> m_velocity
		)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= pos.size()) return;
	
		h[i] = pos[i][1] - solid[i][1];
		m_velocity[i] = Coord(0, 0, 0);
	}

	template<typename Real, typename Coord>
	__global__ void Init_gridVel(
		DeviceArray<Real> grid_vel_x,
		DeviceArray<Real> grid_vel_z
	)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i < grid_vel_x.size()) grid_vel_x[i] = 0;
		if (i < grid_vel_z.size()) grid_vel_z[i] = 0;

	}
	template<typename TDataType>
	ShallowWaterEquationModel<TDataType>::~ShallowWaterEquationModel()
	{
	}

	template<typename TDataType>
	bool ShallowWaterEquationModel<TDataType>::initializeImpl()
	{
		int num = m_position.getElementCount();
		m_accel.setElementCount(num);
		h.setElementCount(num);
		h_buffer.setElementCount(num);
		xcount = num / zcount;
		grid_vel_x.setElementCount((xcount + 1) * (zcount + 2));
		grid_vel_z.setElementCount((xcount + 2) * (zcount + 1));
		grid_accel_x.setElementCount((xcount + 1) * (zcount + 2));
		grid_accel_z.setElementCount((xcount + 2) * (zcount + 1));

		printf("neighbor limit is 4, index count is %d\n", solid.getElementCount());
		cuint pDims = cudaGridSize(num, BLOCK_SIZE);
		cuint pDims2 = cudaGridSize((xcount + 2) * (zcount + 2), BLOCK_SIZE);
		Init <Real, Coord> << < pDims, BLOCK_SIZE >> > (m_position.getValue(), solid.getValue(), h.getValue(), isBound.getValue(), m_velocity.getValue());
		Init_gridVel <Real, Coord> << < pDims2, BLOCK_SIZE >> > (grid_vel_x.getValue(), grid_vel_z.getValue());
		cuSynchronize();
		return true;
	}

	__device__ int neighborFind(int ix, int iz, int j, int zcount) {
		if (j == 0) {
			if (iz == 0)return -1;
			else return ix * zcount + iz - 1;
		}
		else if (j == 1) {
			if (iz + 1 == zcount)return -1;
			else return ix * zcount + iz + 1;
		}
		else if (j == 2) {
			return (ix - 1)*zcount + iz;
		}
		else if (j == 3) {
			return (ix + 1)*zcount + iz;
		}
	}

	template<typename Real, typename Coord>
	__global__ void computeBoundConstrant(
		DeviceArray<Real> h,
		DeviceArray<int> xindex,
		DeviceArray<int> zindex,
		DeviceArray<Coord> m_accel,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_position,
		DeviceArray<int> isBound,
		int zcount,
		Real distance,
		Real gravity,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;

		if (isBound[i] == 0)
			return;
		int maxNei = 4;

		int ix = xindex[i];
		int iz = zindex[i];

		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborFind(ix, iz, j, zcount);
			if (nei >= h.size() || nei < 0)
			{
				switch (j)
				{
				case 0:
				case 1:
					m_velocity[i][2] = 0;
					break;
				case 2:
				case 3:
					m_velocity[i][0] = 0;
					break;
				}
			}

		}
	}

	template<typename Real, typename Coord>
	__global__ void computeAccel(
		DeviceArray<Real> h,
		DeviceArray<int> xindex,
		DeviceArray<int> zindex,
		DeviceArray<Coord> m_accel,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> solid,
		DeviceArray<int> isBound,
		int zcount,
		Real distance,
		Real gravity,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		int maxNei = 4;
		
		int ix = xindex[i];
		int iz = zindex[i];
		Real hx = 0, hz = 0;
		Real p1, p2;
		Real ux = 0, uz = 0, wx = 0, wz = 0;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborFind(ix, iz, j, zcount);
			if (nei >= h.size() || nei < 0)
			{
				continue;
			}
			p1 = m_position[nei][1]; p2 = m_position[i][1];
			if (solid[nei][1] >= m_position[i][1] && h[nei] == 0)
				p1 = p2 = 0;
			if (solid[i][1] >= m_position[nei][1] && h[i] == 0)
				p1 = p2 = 0;

			if(j < maxNei/2)//gradient along z
			{
				hz += (p1 - p2) / (m_position[nei][2] - m_position[i][2]);
				uz += (m_velocity[nei][0] - m_velocity[i][0]) / (m_position[nei][2] - m_position[i][2]);
				wz += (m_velocity[nei][2] - m_velocity[i][2]) / (m_position[nei][2] - m_position[i][2]);
			}
			else
			{
				hx += (p1 - p2) / (m_position[nei][0] - m_position[i][0]);
				ux += (m_velocity[nei][0] - m_velocity[i][0]) / (m_position[nei][0] - m_position[i][0]);
				wx += (m_velocity[nei][2] - m_velocity[i][2]) / (m_position[nei][0] - m_position[i][0]);
			}
		}
		m_accel[i][0] = -gravity * hx / 2;
		m_accel[i][2] = -gravity * hz / 2;

		//m_accel[i][0] = -(gravity * hx + m_velocity[i][0] * ux) / 2 - m_velocity[i][2] * uz / 2;
		//m_accel[i][2] = -(gravity * hz + m_velocity[i][2] * wz) / 2 - m_velocity[i][0] * wx / 2;

		//Real maxAccel = 2 * distance / (dt*dt);
		//Real accel = sqrt(pow(m_accel[i][0], 2) + pow(m_accel[i][2], 2));
		//if(accel > maxAccel)
		//{
		//	printf("%d exceed max accel \n", i);
		//	m_accel[i][0] *= maxAccel / accel;
		//	m_accel[i][2] *= maxAccel / accel;
		//}
	}

	template<typename Real, typename Coord>
	__global__ void computeGridAccel(
		DeviceArray<Real> grid_vel_x,
		DeviceArray<Real> grid_vel_z,
		DeviceArray<Real> grid_accel_x,
		DeviceArray<Real> grid_accel_z,
		DeviceArray<Real> h,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> solid,
		int zcount,
		Real gravity,
		Real distance
	)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		Real hx = 0, hz = 0;
		Real p1, p2;
		Real ux = 0, uz = 0, wx = 0, wz = 0;
		int ix, iz;
		int xcount = m_position.size() / zcount;
		if (i < grid_vel_x.size())
		{
			ix = i / (zcount + 2);
			iz = i % (zcount + 2);
			if (iz == 0 || iz == zcount + 1 || ix == 0 || ix == xcount)
				grid_accel_x[i] = 0;
			else 
			{
				//particles
				int nei1 = (ix - 1) * zcount + iz - 1, nei2 = ix * zcount + iz - 1;
				p1 = m_position[nei1][1]; p2 = m_position[nei2][1];
				if (solid[nei1][1] >= m_position[nei2][1] && h[nei1] == 0)
					p1 = p2 = 0;
				if (solid[nei2][1] >= m_position[nei1][1] && h[nei2] == 0)
					p1 = p2 = 0;
				hx = (p1 - p2) / (m_position[nei1][0] - m_position[nei2][0]);

				//grid
				nei1 = i - 1, nei2 = i + 1;
				uz = (grid_vel_x[nei2] - grid_vel_x[nei1]) / distance * 0.5;
				nei1 = i - zcount - 2; nei2 = i + zcount + 2;
				ux = (grid_vel_x[nei2] - grid_vel_x[nei1]) / distance * 0.5;
				//compute velocity z
				//Real u = grid_vel_x[i], w = 0;
				Real u = grid_vel_x[i] * 0.5 + 
					0.25 * (grid_vel_x[(ix-1)*(zcount+2)+iz] + grid_vel_x[(ix + 1) * (zcount + 2) + iz]),
					w = 0;
				w += grid_vel_z[ix * (zcount + 1) + iz - 1]; w += grid_vel_z[ix * (zcount + 1) + iz];
				w += grid_vel_z[(ix + 1) * (zcount + 1) + iz - 1]; w += grid_vel_z[(ix + 1) * (zcount + 1) + iz];
				w *= 0.25;

				grid_accel_x[i] = -(u * ux + w * uz + gravity * hx);
				//grid_accel_x[i] = -(gravity * hx);
			}
		}
		if (i < grid_vel_z.size())
		{
			ix = i / (zcount + 1);
			iz = i % (zcount + 1);
			if (ix == 0 || iz == 0 || ix == xcount + 1 || iz == zcount)
				grid_accel_z[i] = 0;
			else
			{
				//particles
				int nei1 = (ix - 1) * zcount + iz - 1, nei2 = (ix - 1) * zcount + iz;
				p1 = m_position[nei1][1]; p2 = m_position[nei2][1];
				if (solid[nei1][1] >= m_position[nei2][1] && h[nei1] == 0)
					p1 = p2 = 0;
				if (solid[nei2][1] >= m_position[nei1][1] && h[nei2] == 0)
					p1 = p2 = 0;
				hz = (p1 - p2) / (m_position[nei1][2] - m_position[nei2][2]);

				//grid
				nei1 = i - 1, nei2 = i + 1;
				wz = (grid_vel_z[nei2] - grid_vel_z[nei1]) / distance * 0.5;
				nei1 = i - zcount - 1; nei2 = i + zcount + 1;
				wx = (grid_vel_z[nei2] - grid_vel_z[nei1]) / distance * 0.5;
				//compute velocity z
				//Real u = 0, w = grid_vel_z[i];
				Real u = 0,
					w = 0.5 * grid_vel_z[i] + 0.25 * (grid_vel_z[i - 1] + grid_vel_z[i + 1]);
				u += grid_vel_x[(ix - 1) * (zcount + 2) + iz]; u += grid_vel_x[(ix - 1) * (zcount + 2) + iz + 1];
				u += grid_vel_x[ix * (zcount + 2) + iz]; u += grid_vel_x[ix * (zcount + 2) + iz + 1];
				u *= 0.25;

				grid_accel_z[i] = -(u * wx + w * wz + gravity * hz);
				//grid_accel_z[i] = -gravity * hz;
			}
		}
	}
	template<typename Real, typename Coord>
	__global__ void computeGridVelocity(
		DeviceArray<Real> grid_vel_x,
		DeviceArray<Real> grid_vel_z,
		DeviceArray<Real> grid_accel_x,
		DeviceArray<Real> grid_accel_z,
		int zcount,
		Real relax,
		Real gravity,
		Real distance,
		Real dt
	)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int ix, iz;
		int xcount = grid_vel_x.size() / (zcount + 2) - 1;
		//if (i == 0)
		//	printf("xcount is %d, zcount is %d\n", xcount, zcount);
		if (i < grid_vel_x.size())
		{
			ix = i / (zcount + 2);
			iz = i % (zcount + 2);
			//if (iz == 0 || iz == zcount + 1 || ix == 0 || ix == xcount)
			//	grid_vel_x[i] = 0;
			grid_vel_x[i] = grid_vel_x[i] * relax + grid_accel_x[i] * dt;
		}
		if (i < grid_vel_z.size())
		{
			ix = i / (zcount + 1);
			iz = i % (zcount + 1);
			//if (iz == 0 || iz == zcount || ix == 0 || ix == xcount + 1)
			//	grid_vel_z[i] = 0;
			grid_vel_z[i] = grid_vel_z[i] * relax + grid_accel_z[i] * dt;
		}
		Real maxVel = 2*sqrt(distance * gravity), vel;
		//vel = sqrt(pow(m_velocity[i][0], 2) + pow(m_velocity[i][2], 2));
		vel = abs(grid_vel_x[i]);
		if (vel > maxVel)
		{
			grid_vel_x[i] *= maxVel / vel;
		}
		vel = abs(grid_vel_z[i]);
		if (vel > maxVel)
		{
			grid_vel_z[i] *= maxVel / vel;
		}
	}

	template<typename Real, typename Coord>
	__global__ void computeVelocity(
		DeviceArray<Real> grid_vel_x,
		DeviceArray<Real> grid_vel_z,
		DeviceArray<Real> h,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> m_velocity,
		int zcount,
		Real distance,
		Real gravity,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		int xcount = h.size() / zcount;
		int ix = i / zcount;
		int iz = i % zcount;
		
		m_velocity[i][0] = 0.5 * (grid_vel_x[ix * (zcount + 2) + iz + 1] + grid_vel_x[(ix + 1) * (zcount + 2) + iz + 1]);
		m_velocity[i][2] = 0.5 * (grid_vel_z[(ix + 1) * (zcount + 1) + iz] + grid_vel_z[(ix + 1) * (zcount + 1) + iz + 1]);
		if (ix == 0 || ix == xcount - 1)
			m_velocity[i][0] = 0;
		if (iz == 0 || iz == zcount - 1)
			m_velocity[i][2] = 0;

	}

	template<typename Real, typename Coord>
	__global__ void computeHeight(
		DeviceArray<Real> h,
		DeviceArray<Real> h_buffer,
		DeviceArray<int> xindex,
		DeviceArray<int> zindex,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_accel,
		DeviceArray<int> isBound,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> solid,
		DeviceArray<Coord> normal,
		int zcount,
		Real distance,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		int maxNei = 4;
		int ix = xindex[i];
		int iz = zindex[i];

		Real uhx = 0, whz = 0;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborFind(ix, iz, j, zcount);
			//bound cell
			if (nei >= h.size() || nei < 0)
			{
				continue;
			}
			if (j < maxNei / 2)//gradient along z
			{
				whz += (h[nei] * m_velocity[nei][2] - h[i] * m_velocity[i][2]) / (m_position[nei][2] - m_position[i][2]);
			}
			else
			{
				uhx += (h[nei] * m_velocity[nei][0] - h[i] * m_velocity[i][0]) / (m_position[nei][0] - m_position[i][0]);
			}
		}
		h_buffer[i] = -(uhx / 2 + whz / 2)*dt;
	}

	template<typename Real, typename Coord>
	__global__ void applyHeight(
		DeviceArray<Real> h,
		DeviceArray<Real> h_buffer,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> solid,
		DeviceArray<Coord> m_velocity
	)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		h[i] += h_buffer[i];
		if (h[i] < 0)
		{
			h[i] = 0;
			//m_velocity[i][0] = m_velocity[i][2] = 0;
			m_velocity[i][1] = max(0.0, m_velocity[i][1]);
		}
		m_position[i][1] = solid[i][1] + h[i];
	}
	template<typename TDataType>
	void ShallowWaterEquationModel<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}

		int num = m_position.getElementCount();
		cuint pDims = cudaGridSize(num, BLOCK_SIZE);
		cuint pDims2 = cudaGridSize(max((zcount+1)*(xcount+2), (zcount+2)*(xcount+1)), BLOCK_SIZE);

		////计时开始
		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start);

		computeGridAccel <Real, Coord> << < pDims2, BLOCK_SIZE >> > (
			grid_vel_x.getValue(),
			grid_vel_z.getValue(),
			grid_accel_x.getValue(),
			grid_accel_z.getValue(),
			h.getValue(),
			m_position.getValue(),
			solid.getValue(),
			zcount,
			9.8,
			distance
			);
		cuSynchronize();
		cudaDeviceSynchronize();

		computeGridVelocity <Real, Coord> << < pDims2, BLOCK_SIZE >> > (
			grid_vel_x.getValue(),
			grid_vel_z.getValue(),
			grid_accel_x.getValue(),
			grid_accel_z.getValue(),
			zcount,
			relax,
			9.8,
			distance,
			dt
			);
		cuSynchronize();
		cudaDeviceSynchronize();

		computeVelocity <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			grid_vel_x.getValue(),
			grid_vel_z.getValue(),
			h.getValue(),
			m_position.getValue(),
			m_velocity.getValue(),
			zcount,
			distance,
			9.8,
			dt
			);
		cuSynchronize();
		cudaDeviceSynchronize();

		computeHeight <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			h_buffer.getValue(),
			xindex.getValue(),
			zindex.getValue(),
			m_velocity.getValue(),
			m_accel.getValue(),
			isBound.getValue(),
			m_position.getValue(),
			solid.getValue(),
			normal.getValue(),
			zcount,
			distance,
			dt
			);
		cuSynchronize();
		cudaDeviceSynchronize();

		applyHeight <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			h_buffer.getValue(),
			m_position.getValue(),
			solid.getValue(),
			m_velocity.getValue()
			);
		cuSynchronize();
		cudaDeviceSynchronize();

		//cudaEventRecord(stop);

		//cudaEventSynchronize(stop);
		//float milliseconds = 0;
		//cudaEventElapsedTime(&milliseconds, start, stop);

		//sumtimes += milliseconds;
		//sumnum++;
		//printf("Time: %f \n", sumtimes / sumnum);
		/*
		//输出看看
		if (sumnum == 1000) {
			std::string s = "C:\\temp.txt";
			std::ofstream myout;
			myout.open(s, std::ios::out | std::ios::app);
			if (!myout) {
				std::ofstream newout(s);
				newout << sumtimes / sumnum << std::endl;
			}
			myout << sumtimes / sumnum << std::endl;
			myout.close();
		}
		*/
	/*	cublasHandle_t handle;
		float sum;
		cublasCreate(&handle);
		cublasSasum(handle, solid.getElementCount(), h.getValue().getDataPtr(), 1, &sum);
		cublasDestroy(handle);
		printf("total height is %f\n", sum);*/
	}
}
