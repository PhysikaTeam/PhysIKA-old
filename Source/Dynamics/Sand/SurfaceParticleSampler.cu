//#include "SurfaceParticleSampler.h"
//
//#include "Core/Utility/cuda_utilities.h"
//#include "Core/Utility/CudaRand.h"
//
//namespace PhysIKA
//{
//	SurfaceParticleSampler::SurfaceParticleSampler()
//	{
//	}
//
//	SurfaceParticleSampler::~SurfaceParticleSampler()
//	{
//		m_normalizePosition.Release();
//		//bExist.Release();
//	}
//
//	void SurfaceParticleSampler::Initalize(int nx, int ny, int layer, float gridLength)
//	{
//		m_nx = nx;
//		m_ny = ny;
//		m_sampleLayer = layer;
//		m_gridLength = gridLength;
//
//		m_normalizePosition.Resize(m_nx, m_ny, m_sampleLayer);
//	}
//
//	__global__ void g_generateParticles(Grid3f position)
//	{
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//		int nx = position.Nx();
//		int ny = position.Ny();
//		int layer = position.Nz();
//
//		if (i >= nx) return;
//		if (j >= ny) return;
//
//		RandNumber gen(position.Index(i, j, 0));
//
//		for (int k = 0; k < layer; ++k)
//		{
//			float x, y, z;
//			x = gen.Generate();
//			y = gen.Generate();
//			z = gen.Generate();
//
//			position(i, j, k) = make_float3(x, y, z);
//		}
//	}
//
//	void SurfaceParticleSampler::Generate()
//	{
//		dim3 bDims(16,16,1);
//		dim3 gDims(cudaGridSize(m_nx, bDims.x), cudaGridSize(m_ny, bDims.y), 1);
//
//		g_generateParticles << <gDims, bDims >> > (m_normalizePosition);
//	}
//
//
//	__host__ __device__ vertex sampToVertex(float3 gp, float x, float y, float h, int Nx, int Ny, float dl)
//	{
//		vertex v;
//
//		if (gp.x < 0 || gp.y < 0 || gp.z < 0)
//		{
//			v.x = 0;
//			v.z = 0;
//			v.y = 0;
//		}
//		else
//		{
//			v.x = (x/* - Nx / 2*/)* dl;
//			v.y = (y/* - Ny / 2*/)* dl;
//			v.z = -gp.z + h;// +100 * dl;
//		}
//
//		//v = v * 0.05f;
//
//		return v;
//	}
//
//	__global__ void d_visualize_samples_2(float4* samples, /*rgb* sample_color,*/ Grid3f position, int width, int height, float sample_spacing, float grid_spacing)
//	{
//		int i = threadIdx.x + blockIdx.x * blockDim.x;
//		int j = threadIdx.y + blockIdx.y * blockDim.y;
//		if (i >= position.nx || j >= position.ny)
//			return;
//
//
//		for (int k = 0; k < position.nz; k++)
//		{
//			float3 pos = position(i, j, k);
//			float grid_fx = (i + pos.x)*sample_spacing;
//			float grid_fy = (j + pos.y)*sample_spacing;
//
//			if (grid_fx < 0.0f) grid_fx = 0.0f;
//			if (grid_fx > width - 1) grid_fx = width - 1.0f;
//			if (grid_fy < 0.0f) grid_fy = 0.0f;
//			if (grid_fy > height - 1) grid_fy = height - 1.0f;
//
//			int gridx = floor(grid_fx);		int gridy = floor(grid_fy);
//			float fx = grid_fx - gridx;		float fy = grid_fy - gridy;
//
//			if (gridx == width - 1) { gridx = width - 2; fx = 1.0f; }
//			if (gridy == height - 1) { gridy = height - 2; fy = 1.0f; }
//
//			float w00 = (1.0f - fx)*(1.0f - fy);
//			float w10 = fx * (1.0f - fy);
//			float w01 = (1.0f - fx)*fy;
//			float w11 = fx * fy;
//
//			float4 gp00 = tex2D(texture_grid, gridx + 1, gridy + 1);
//			float4 gp10 = tex2D(texture_grid, gridx + 2, gridy + 1);
//			float4 gp01 = tex2D(texture_grid, gridx + 1, gridy + 2);
//			float4 gp11 = tex2D(texture_grid, gridx + 2, gridy + 2);
//
//			float4 gp = w00 * gp00 + w10 * gp10 + w01 * gp01 + w11 * gp11;
//			/*printf("gridx=%d,gridy=%d,gp00.w=%f\n", gridx, gridy, gp00.w);*/
//			samples[position.Index(i, j, k)] = sampToVertex(make_float3(grid_fx, grid_fy, k*sample_spacing), grid_fx, grid_fy, gp.x + gp.w, width, height, grid_spacing);
//			//sample_color[position.Index(i, j, k)] = make_uchar4(255, 255, 122, 255);
//		}
//	}
//
//	//__device__ float2 d_TransferVelocity(float4 vel)
//	//{
//	//	float u = vel.x < EPSILON ? 0.0f : vel.y / vel.x;
//	//	float v = vel.x < EPSILON ? 0.0f : vel.z / vel.x;
//	//	return make_float2(u, v);
//	//}
//
//	//__global__ void K_AdvectParticles(Grid3f p_pos, float p_spacing, float4* g_vel, int g_nx, int g_ny, int pitch, float g_spacing, float dt)
//	//{
//	//	uint i = blockDim.x * blockIdx.x + threadIdx.x;
//	//	uint j = blockIdx.y * blockDim.y + threadIdx.y;
//	//	uint k = blockIdx.z * blockDim.z + threadIdx.z;
//
//	//	int p_nx = p_pos.nx;
//	//	int p_ny = p_pos.ny;
//	//	int p_nz = p_pos.nz;
//
//	//	if (i >= p_nx) return;
//	//	if (j >= p_ny) return;
//	//	if (k >= p_nz) return;
//
//	//	int k0 = p_pos.Index(i, j, k);
//
//	//	float w00, w10, w01, w11;
//	//	int g_ix, g_iy, g_iz;
//
//	//	float3 p_ijk = p_pos[k0];
//
//	//	float g_fx = (i + p_ijk.x) * p_spacing / g_spacing;
//	//	float g_fy = (j + p_ijk.y) * p_spacing / g_spacing;
//
//	//	if (g_fx < 0.0f) g_fx = 0.0f;
//	//	if (g_fx > g_nx - 1) g_fx = g_nx - 1.0f;
//	//	if (g_fy < 0.0f) g_fy = 0.0f;
//	//	if (g_fy > g_ny - 1) g_fy = g_ny - 1.0f;
//
//	//	g_ix = floor(g_fx);		g_iy = floor(g_fy);
//	//	g_fx -= g_ix;			g_fy -= g_iy;
//
//	//	if (g_ix == g_nx - 1) { g_ix = g_nx - 2; g_fx = 1.0f; }
//	//	if (g_iy == g_ny - 1) { g_iy = g_ny - 2; g_fy = 1.0f; }
//
//	//	w00 = (1.0f - g_fx)*(1.0f - g_fy);
//	//	w10 = g_fx * (1.0f - g_fy);
//	//	w01 = (1.0f - g_fx)*g_fy;
//	//	w11 = g_fx * g_fy;
//
//	//	g_ix++;
//	//	g_iy++;
//
//	//	// 	if (i + 1 < g_nx*0.45 + 8)
//	//	// 	{
//	//	// 		gridpoint pt = grid2Dread(g_vel, i + 1, j + 1, g_nx);
//	//	// 		pt.y = 0.0f;
//	//	// 		pt.z = 0.0f;
//	//	// 		grid2Dwrite(g_vel, i + 1, j + 1, g_nx, pt);
//	//	// 	}
//
//	//	float2 vel_ijk = w00 * d_TransferVelocity(grid2Dread(g_vel, g_ix, g_iy, pitch)) + w10 * d_TransferVelocity(grid2Dread(g_vel, g_ix + 1, g_iy, pitch)) + w01 * d_TransferVelocity(grid2Dread(g_vel, g_ix, g_iy + 1, pitch)) + w11 * d_TransferVelocity(grid2Dread(g_vel, g_ix + 1, g_iy + 1, pitch));
//
//	//	// 	if (g_ix < 20)
//	//	// 	{
//	//	// // 		float2 vel_tmp = d_TransferVelocity(grid2Dread(g_vel, g_ix, g_iy, g_nx));
//	//	// // 		if (abs(vel_tmp.y) > EPSILON)
//	//	// // 		{
//	//	// // 			printf("%i %i \n", g_ix, g_iy);
//	//	// // 		}
//	//	//
//	//	// 		vel_ijk.x = 0.0f;
//	//	// 		vel_ijk.y = 0.0f;
//	//	// 	}
//
//	//	p_ijk.x += vel_ijk.x * dt / p_spacing;
//	//	p_ijk.y += vel_ijk.y * dt / p_spacing;
//
//	//	p_pos(i, j, k) = p_ijk;
//	//}
//
//
//	//void SurfaceParticleSampler::Advect(float4* g_vel, int g_nx, int g_ny, int pitch, float g_spacing, float dt)
//	//{
//	//	dim3 gridDims, blockDims;
//	//	uint3 fDims = make_uint3(Nx, Ny, Nz);
//	//	computeGridSize3D(fDims, make_uint3(8, 8, 1), gridDims, blockDims);
//	//	K_AdvectParticles << < gridDims, blockDims >> > (position, m_spacing, g_vel, g_nx, g_ny, pitch, g_spacing, dt);
//	//}
//
//
//	//__global__ void g_DepositPigments(Grid3f p_pos, Grid3f prePos, Grid1b bExist, Grid1b pre_bExist, Grid1u gMutex)
//	//{
//	//	uint i = blockIdx.x * blockDim.x + threadIdx.x;
//	//	uint j = blockIdx.y * blockDim.y + threadIdx.y;
//	//	uint k = blockIdx.z * blockDim.z + threadIdx.z;
//
//	//	int nx = p_pos.nx;
//	//	int ny = p_pos.ny;
//	//	int nz = p_pos.nz;
//
//	//	if (i >= nx) return;
//	//	if (j >= ny) return;
//	//	if (k >= nz) return;
//
//	//	if (!pre_bExist(i, j, k)) return;
//
//	//	int ix, iy, iz;
//	//	float fx, fy, fz;
//
//	//	int id = prePos.Index(i, j, k);
//	//	float3 pos = prePos[id];
//
//	//	ix = floor(i + pos.x);
//	//	iy = floor(j + pos.y);
//	//	iz = floor(k + pos.z);
//
//	//	fx = i + pos.x - ix;
//	//	fy = j + pos.y - iy;
//	//	fz = k + pos.z - iz;
//
//	//	float3 p = make_float3(fx, fy, fz);
//
//	//	if (ix < 0) { return; }
//	//	if (ix >= nx) { return; }
//	//	if (iy < 0) { return; }
//	//	if (iy >= ny) { return; }
//	//	if (iz < 0) { return; }
//	//	if (iz >= nz) { return; }
//
//	//	int id_new = p_pos.Index(ix, iy, iz);
//
//	//	while (atomicCAS(&(gMutex[id_new]), 0, 1) == 0) break;
//	//	p_pos[id_new] = p;
//	//	bExist[id_new] = true;
//	//	atomicExch(&(gMutex[id_new]), 0);
//	//}
//
//	//void SurfaceParticleSampler::Deposit()
//	//{
//	//	dim3 gridDims, blockDims;
//	//	uint3 fDims = make_uint3(Nx, Ny, Nz);
//	//	computeGridSize3D(fDims, make_uint3(8, 8, 1), gridDims, blockDims);
//
//	//	pre_position.CopyFrom(position);
//	//	pre_bExist.CopyFrom(bExist);
//
//	//	position.Clear();
//	//	bExist.Clear();
//
//	//	g_DepositPigments << < gridDims, blockDims >> > (position, pre_position, bExist, pre_bExist, m_mutex);
//	//}
//
//}