#include "SSESandSolver.h"

#include <cuda_runtime.h>
#include <curand_kernel.h> 
#include "SandGrid.h"
#include "SSEUtil.h"
#include "Core/Utility/Reduction.h"
#include "Core/Utility/cuda_utilities.h"
//#include <cooperative_groups.h>
//using namespace cooperative_groups;


namespace PhysIKA
{
	using namespace SSEUtil;

	extern texture<float4, 2, cudaReadModeElementType> texture_grid;
	//extern texture<float, 2, cudaReadModeElementType> texture_land;

	SSESandSolver::SSESandSolver()
	{
		//m_CFLReduction = Reduction<float>::Create();
	}

	SSESandSolver::~SSESandSolver()
	{
		if (m_CFLReduction)
			delete m_CFLReduction;

	}

	bool SSESandSolver::initialize()
	{
		m_sandData.updateLandGridHeight();
		m_sandData.getSandGridInfo(m_SandInfo);
		//m_pSandInfo = &m_sandinfo;
		this->setSandGridInfo(m_SandInfo);

		if (m_sandStaticHeight.Nx() != m_sandData.Nx || m_sandStaticHeight.Ny() != m_sandData.Ny)
		{
			m_sandStaticHeight.resize(m_sandData.Nx, m_sandData.Ny);
			m_sandStaticHeight.Reset();
			//Function1Pt::copy(m_sandStaticHeight, m_sandData.m_sandHeight);

			Vector3d ori = m_sandData.m_sandHeight.getOrigin();
			m_sandStaticHeight.setOrigin(ori[0], ori[1], ori[2]);
			m_sandStaticHeight.setSpace(m_sandData.m_sandHeight.getDx(), m_sandData.m_sandHeight.getDz());

			m_macStaticHeightx.resize(m_sandData.Nx + 1, m_sandData.Ny);
			m_macStaticHeightx.Reset();
			m_macStaticHeightx.setOrigin(ori[0], ori[1], ori[2]);
			m_macStaticHeightx.setSpace(m_sandData.m_sandHeight.getDx(), m_sandData.m_sandHeight.getDz());

			m_macStaticHeightz.resize(m_sandData.Nx, m_sandData.Ny + 1);
			m_macStaticHeightz.Reset();
			m_macStaticHeightz.setOrigin(ori[0], ori[1], ori[2]);
			m_macStaticHeightz.setSpace(m_sandData.m_sandHeight.getDx(), m_sandData.m_sandHeight.getDz());
		}

		for(int i=0;i<1;++i)
			this->updateSandStaticHeight(0.016);

		return true;
	}

	bool SSESandSolver::stepSimulation(float deltime)
	{
		do {
			double subdt = this->getMaxTimeStep();
			subdt = subdt < deltime ? subdt : deltime;
			deltime -= subdt;
			printf("  Cur time step:  %f\n", subdt);

			this->advection(subdt);


			this->updateVeclocity(subdt);

			

		} while (deltime > 0);
		

		m_sandData.updateSandGridHeight();

		//this->updateSandStaticHeight(deltime);

		return true;
	}


	__global__ void g_sandAdvection(float4* grid_next, int width, int height, float timestep, int pitch, float griddl)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{

			int gridx = x;
			int gridy = y;

			float4 center = tex2D(texture_grid, gridx, gridy);
			float4 north = tex2D(texture_grid, gridx, max(0, gridy - 1));
			float4 west = tex2D(texture_grid, max(0, gridx - 1), gridy);
			float4 south = tex2D(texture_grid, gridx, min(height - 1, gridy + 1));
			float4 east = tex2D(texture_grid, min(width - 1, gridx + 1), gridy);

			float4 eastflux = d_flux_v(center, east);
			float4 westflux = d_flux_v(west, center);
			float4 southflux = d_flux_u(center, south);
			float4 northflux = d_flux_u(north, center);
			float4 flux = eastflux - westflux + southflux - northflux;
			float4 u_center = center - (timestep / griddl) *flux;



			if (u_center.x < EPSILON)
			{
				u_center.x = 0.0f;
				u_center.y = 0.0f;
				u_center.z = 0.0f;
			}

			float totalH = u_center.x + center.w;
			if ((east.w >= totalH) || (west.w >= totalH))
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			if ((north.w >= totalH) || (south.w >= totalH))
			{
				u_center.y = 0;
				//u_center.z = 0;
			}

			// boundary condition: vn = 0
			if (x == 0 || x == width - 1)
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			if (y == 0 || y == height - 1)
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			u_center.w = center.w;

			//if (u_center.x > 0.21)
			//{
			//	printf("  Advection error:  %lf %lf %lf,  before:  %lf %lf %lf,   flux: %lf %lf %lf\n", u_center.x, u_center.y, u_center.z,
			//		center.x, center.y, center.z, flux.x, flux.y, flux.z);
			//}

			grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
		}
	}

	void SSESandSolver::advection(float deltime)
	{
		dim3 gridDims, blockDims;
		int in_Nx = m_SandInfo.nx, in_Ny = m_SandInfo.ny;
		make_dimension2D(in_Nx, in_Ny, m_threadBlockx, m_threadBlocky, gridDims, blockDims);

	
		// bind texture.
		SSEUtil::bindTexture2D(texture_grid, m_SandInfo.data, m_SandInfo.nx, m_SandInfo.ny, m_SandInfo.pitch);

		g_sandAdvection << < gridDims, blockDims >> > (m_SandInfo.data, in_Nx, in_Ny,
			deltime, m_SandInfo.pitch, m_SandInfo.griddl);
		// cudaThreadSynchronize();
		cuSynchronize();

		// bind texture
		//SSEUtil::bindTexture2D(texture_grid, m_SandInfo.data, m_SandInfo.nx, m_SandInfo.ny, m_SandInfo.pitch);
		//cuSynchronize();
	}


	/**
	*@brief Apply pressure and friction to sand.
	*@details The direction notation may be a little confusing.
	*@details Remember that EAST&WEST is (x, v, width) direction in global, and SOUTH&NORTH is (z, u, height) direction.
	*/
	__global__ void g_updateVelocity(DeviceHeightField1d staticHeight, float4* grid_next, int width, int height, float timestep, int pitch, float sliding_depth, float drag_force, float mu, float dl)
	{

		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			int gridx = x;
			int gridy = y;

			float4 center = tex2D(texture_grid, gridx, gridy);
			float4 north = tex2D(texture_grid, gridx, max(0, gridy - 1));
			float4 west = tex2D(texture_grid, max(0, gridx - 1), gridy);
			float4 south = tex2D(texture_grid, gridx, min(height - 1, gridy + 1));
			float4 east = tex2D(texture_grid, min(width - 1, gridx + 1), gridy);

			float h = center.x;
			float hu_new = center.y;
			float hv_new = center.z;

			float hu_old = hu_new;
			float hv_old = hv_new;

			float s = center.x + center.w;
			float sw = west.x + west.w;
			float se = east.x + east.w;
			float sn = north.x + north.w;
			float ss = south.x + south.w;


			//float  h_d = center.x - staticHeight(x, y);//sliding_depth;
			float  h_d = sliding_depth;


			float2 sliding_dir;
			sliding_dir.y = (sw - se) / (2.0 * dl);
			sliding_dir.x = (sn - ss) / (2.0 * dl);
			float gradient = sqrtf(sliding_dir.x*sliding_dir.x + sliding_dir.y*sliding_dir.y);

			float sliding_cos = 1 / sqrtf(1 + gradient * gradient);
			float sliding_sin = abs(gradient) / sqrtf(1 + gradient * gradient);

			float2 hvel_old = make_float2(center.y, center.z);
			float2 hvel_new;

			float sliding_length = sqrtf(sliding_dir.x*sliding_dir.x + sliding_dir.y*sliding_dir.y);

			// apply pressure
			float g = GRAVITY;
			float hu_tmp = hu_old + timestep * g*fmaxf(fminf(h_d, center.x), 0)*sliding_dir.x;
			float hv_tmp = hv_old + timestep * g*fmaxf(fminf(h_d, center.x), 0)*sliding_dir.y;

			float2 vel_dir;
			float vel_norm = sqrtf(hu_tmp * hu_tmp + hv_tmp * hv_tmp);
			if (vel_norm < EPSILON)
			{
				vel_dir.x = 0.0f;
				vel_dir.y = 0.0f;
			}
			else
			{
				vel_dir.x = hu_tmp / vel_norm;
				vel_dir.y = hv_tmp / vel_norm;
			}

			// apply bottom friction
			hu_new = hu_tmp - timestep * g*fmaxf(fminf(h_d, center.x), 0)*vel_dir.x*mu;
			hv_new = hv_tmp - timestep * g*fmaxf(fminf(h_d, center.x), 0)*vel_dir.y*mu;

			if (hu_new*hu_tmp + hv_new * hv_tmp < EPSILON && sliding_sin - mu * sliding_cos < 0)
			{
				hu_new = 0.0f;
				hv_new = 0.0f;
			}


			float4 u_center;
			u_center.x = center.x;
			u_center.y = hu_new * drag_force;
			u_center.z = hv_new * drag_force;
			float totalH = u_center.x + center.w;
			if (u_center.x <= EPSILON)
			{
				//u_center.x = 0;
				u_center.y = 0;
				u_center.z = 0;
			}
			if ((east.w >= totalH) || (west.w >= totalH))
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			if ((north.w >= totalH) || (south.w >= totalH))
			{
				u_center.y = 0;
				//u_center.z = 0;
			}

			if (x == 0 || x == width - 1)
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			if (y == 0 || y == height - 1)
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			u_center.w = center.w;

			//if (abs(u_center.y) > 1 || abs(u_center.z) > 1)
			//{
			//	printf("   May be error:  %lf %lf %lf,   befor:  %lf %lf %lf \n", u_center.x, u_center.y, u_center.z,
			//		center.x, center.y, center.z);
			//}

			grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
		}
	}

	void SSESandSolver::updateVeclocity(float deltime)
	{
		dim3 gridDims, blockDims;
		int in_Nx = m_SandInfo.nx, in_Ny = m_SandInfo.ny;
		make_dimension2D(in_Nx, in_Ny, m_threadBlockx, m_threadBlocky, gridDims, blockDims);
		
		// bind texture
		SSEUtil::bindTexture2D(texture_grid, m_SandInfo.data, m_SandInfo.nx, m_SandInfo.ny, m_SandInfo.pitch);

		g_updateVelocity << < gridDims, blockDims >> > (m_sandStaticHeight,
			m_SandInfo.data, in_Nx, in_Ny,
			deltime, m_SandInfo.pitch, m_SandInfo.slide, m_SandInfo.drag, m_SandInfo.mu, m_SandInfo.griddl);
		//cudaThreadSynchronize();
		cuSynchronize();

		// bind texture
		//SSEUtil::bindTexture2D(texture_grid, m_SandInfo.data, m_SandInfo.nx, m_SandInfo.ny, m_SandInfo.pitch);
		//cuSynchronize();

	}


	__global__ void SSESand_updateMacStaticHeight(
		DeviceHeightField1d macStaticHeight,
		DeviceHeightField1d staticHeight,
		DeviceHeightField1d sandHeight,
		DeviceHeightField1d landHeight,
		double gamma, double threshold, double mu, double dt
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= macStaticHeight.Nx() || y >= macStaticHeight.Ny()) return;

		int ix = -1, iy = 0;

		Vector3d pos = macStaticHeight.gridCenterPosition(x, y);
		double statich = staticHeight.get(pos[0], pos[2]);
		

		Vector3d grads;
		staticHeight.gradient(pos[0], pos[2], grads[0], grads[2]);
		Vector3d gradl;
		landHeight.gradient(pos[0], pos[2], gradl[0], gradl[2]);

		double sandh = sandHeight.get(pos[0], pos[2]);  
		double landh = landHeight.get(pos[0], pos[2]);
		double theta = (grads + gradl).norm();
		double epsilon = gamma * (sandh - statich) * (theta - mu);
		statich = statich - epsilon * dt;
		statich = (sandh - threshold) < statich ? (sandh - threshold) : statich;
		statich = 0 > statich ? 0 : statich;
		macStaticHeight(x, y) = statich;

		//double sandh = sandHeight(x, y);
		//double flowh = sandh - staticHeight(x, y);

		//Vector3d grad1;
		//Vector3d pos = macStaticHeight.gridCenterPosition(x, y);
		//macStaticHeight.gradient(pos[0], pos[2], grad1[0], grad1[2]);

		//Vector3d grad2;
		//pos = landHeight.gridCenterPosition(x, y);
		//landHeight.gradient(pos[0], pos[2], grad2[0], grad2[2]);
		//double theta = (grad1 + grad2).norm();
		//double epsilon = gamma * flowh * (theta - mu);
		//staticHeight(x, y) -= epsilon * dt;

		//flowh = sandh - staticHeight(x, y);
		//if (flowh < threshold)
		//{
		//	flowh = threshold > sandh ? sandh : threshold;
		//	staticHeight(x, y) = sandh - flowh;
		//}
		
		//double epsilon = gamma * 
	}

	__global__ void SSESand_updateStaticHeight(
		DeviceHeightField1d staticHeight,
		DeviceHeightField1d macStaticHeightx,
		DeviceHeightField1d macStaticHeightz,
		DeviceHeightField1d sandHeight,
		DeviceHeightField1d landHeight,
		double threshold
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= staticHeight.Nx() && y >= staticHeight.Ny()) return;

		Vector3d pos = staticHeight.gridCenterPosition(x, y);
		double statichx = (macStaticHeightx(x, y) + macStaticHeightx(x + 1, y)) / 2.0;
		double statichz = (macStaticHeightz(x, y) + macStaticHeightz(x, y + 1)) / 2.0;
		double statich = (statichx + statichz) / 2.0;

		double sandh = sandHeight(x, y);
		statich = (sandh - threshold) < statich ? (sandh - threshold) : statich;
		statich = 0 > statich ? 0 : statich;
		staticHeight(x, y) = statich;
	}

	void SSESandSolver::updateSandStaticHeight(float dt)
	{
		//return;

		uint3 gsize = { m_sandStaticHeight.Nx(), m_sandStaticHeight.Ny(), 1 };
		double threshold = 0.0;

		gsize.x = m_macStaticHeightx.Nx();
		gsize.y = m_macStaticHeightx.Ny();
		cuExecute2D(gsize, SSESand_updateMacStaticHeight,
			m_macStaticHeightx,
			m_sandStaticHeight,
			m_sandData.m_sandHeight,
			m_sandData.m_landHeight,
			1.0, threshold, 1.0, dt
		);

		gsize.x = m_macStaticHeightz.Nx();
		gsize.y = m_macStaticHeightz.Ny();
		cuExecute2D(gsize, SSESand_updateMacStaticHeight,
			m_macStaticHeightz,
			m_sandStaticHeight,
			m_sandData.m_sandHeight,
			m_sandData.m_landHeight,
			1.0, threshold, 1.0, dt
		);



		gsize.x = m_sandStaticHeight.Nx();
		gsize.y = m_sandStaticHeight.Ny();
		cuExecute2D(gsize, SSESand_updateStaticHeight,
			m_sandStaticHeight,
			m_macStaticHeightx,
			m_macStaticHeightz,
			m_sandData.m_sandHeight,
			m_sandData.m_landHeight,
			threshold
		);
	}


	void SSESandSolver::setSandGridInfo(const SandGridInfo& sandinfo)
	{
		m_SandInfo = sandinfo;

		m_velocityNorm.resize(sandinfo.nx * sandinfo.ny);
	}


	//__global__ void get_max_velocity(gridpoint* grid, int pitch, int width, int height, float* maxV)
	//{
	//	int x = threadIdx.x + blockIdx.x * blockDim.x;
	//	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//	if (x < width && y < height)
	//	{
	//		int gridx = x + 1;
	//		int gridy = y + 1;
	//		gridpoint gp = grid[gridx + gridy * pitch];
	//		float u = d_get_u(gp);
	//		float v = d_get_v(gp);

	//		atomicMax(maxV, abs(u*u + v * v));
	//	}
	//}

	__global__ void g_getVelocityNorm(float* velocityNorm, int width, int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width &&y < height)
		{
			gridpoint gp = tex2D(texture_grid, x, y);
			float velU = d_get_u(gp);
			float velV = d_get_v(gp);

			grid2Dwrite(velocityNorm, x, y, width, sqrtf(velU*velU + velV * velV));
		}
	}

//	__global__ void testfun23423(float a, float b)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//	grid_group grid = this_grid();
//	printf("%d:  %f\n", tid, a);
//
//	//grid.sync();
//	printf("%d:  %f   %f\n", tid, b, b);
//}

	float SSESandSolver::getMaxTimeStep()
	{
		int ngrid = m_SandInfo.nx * m_SandInfo.ny;
		if (!m_CFLReduction)
			m_CFLReduction = Reduction<float>::Create(ngrid);
		
		// bind texture
		SSEUtil::bindTexture2D(texture_grid, m_SandInfo.data, m_SandInfo.nx, m_SandInfo.ny, m_SandInfo.pitch);

		// Veclocity values of sand.
		dim3 bdim(16, 16, 1);
		dim3 gdim = cudaGridSize3D(dim3(m_SandInfo.nx, m_SandInfo.ny, 1), bdim);
		g_getVelocityNorm << <gdim, bdim >> > (m_velocityNorm.begin(), m_SandInfo.nx, m_SandInfo.ny);
		cuSynchronize();

		// Maximum velocity.
		float maxVel = m_CFLReduction->maximum(m_velocityNorm.begin(), ngrid);

		// Maximum time step.
		float maxDl = m_CFLNumber * m_SandInfo.griddl;
		float dt = m_maxTimeStep;
		dt = (maxDl > dt*maxVel) ? dt : maxDl / maxVel;
		
		return dt;
	}


	__global__ void SSESand_applyVelocityChange(
		SandGridInfo sandinfo,
		DeviceDArray<Vector3d> gridVel,
		int minGi, int minGj, int sizeGi, int sizeGj
	)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		if (tid >= gridVel.size())return;

		//int gi = tid % sizeGi + minGi;
		//int gj = tid / sizeGi + minGj;
		int lgi = 0, lgj = 0;
		SSEUtil::idx2Grid(lgi, lgj, tid, sizeGi);
		int gi = lgi + minGi;
		int gj = lgj + minGj;

		//SSEUtil::idx2Grid(gi, gj, tid, sandinfo.nx);

		
		float w_r = 0;
		if (SSEUtil::inRange(lgi + 1, lgj, sizeGi, sizeGj))
		{
			int idx_r = SSEUtil::grid2Idx(lgi + 1, lgj, sizeGi);
			w_r = gridVel[idx_r][1];
		}

		float w_l = 0;
		if (SSEUtil::inRange(lgi - 1, lgj, sizeGi, sizeGj))
		{
			int idx_l = SSEUtil::grid2Idx(lgi - 1, lgj, sizeGi);
			w_l = gridVel[idx_l][1];
		}
		
		float w_u = 0;
		if (SSEUtil::inRange(lgi, lgj - 1, sizeGi, sizeGj))
		{
			int idx_u = SSEUtil::grid2Idx(lgi, lgj - 1, sizeGi);
			w_u = gridVel[idx_u][1];
		}

		float w_d = 0;
		if (SSEUtil::inRange(lgi, lgj + 1, sizeGi, sizeGj))
		{
			int idx_d = SSEUtil::grid2Idx(lgi, lgj + 1, sizeGi);
			w_d = gridVel[idx_d][1];
		}

		gridpoint gp = grid2Dread(sandinfo.data, gi, gj, sandinfo.pitch);
		
		gp.y = gridVel[tid][2] * gp.x;
		gp.z = gridVel[tid][0] * gp.x;

		double dhu = sandinfo.griddl / 2.0 * (w_r - w_l);
		double dhv = sandinfo.griddl / 2.0 * (w_d - w_u);
		gp.y += dhv;
		gp.z += dhu;

		//if (w_r != 0 || w_l != 0 || w_d != 0 || w_u != 0)
		//{
		//	printf("  Nono 0 vel:  %lf %lf %lf\n", w_r, w_l, w_d, w_u);
		//}

		//if (gridVel[tid][0] != 0 || gridVel[tid][2] != 0)
		//{
		//	printf("gridVel")
		//}

		//if (dhu != 0 || dhv != 0)
		//{
		//	printf("DHUV: %d %d,  %lf %lf %lf %lf\n",gi, gj, dhu, dhv, w_d, w_u);
		//}

		grid2Dwrite(sandinfo.data, gi, gj, sandinfo.pitch, gp);
	}


	void SSESandSolver::applyVelocityChange(float dt, int minGi, int minGj, int sizeGi, int sizeGj)
	{
		if (!m_gridVel)return;




		cuExecute(m_gridVel->size(), SSESand_applyVelocityChange,
			m_SandInfo,
			*m_gridVel,
			minGi, minGj,sizeGi, sizeGj
		);

	}

}