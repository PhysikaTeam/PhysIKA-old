#include "Dynamics/Sand/SandVisualPointSampleModule.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/count.h>

#include "Framework/Topology/PointSet.h"
#include "Core/Typedef.h"

#include "Dynamics/HeightField/HeightFieldGrid.h"

#include "Dynamics/Sand/types.h"

#include "Framework/Framework/Node.h"
#include "Core/Utility/cuda_utilities.h"
//#include "Dynamics/Sand/SSEUtil.h"

//#include <cuda_runtime.h>
//#ifndef __CUDACC__  
//#define __CUDACC__  
//#include "cuda_texture_types.h"  
//#endif 

namespace PhysIKA
{
	

	SandHeightRenderParticleSampler::SandHeightRenderParticleSampler()
	{
	}

	SandHeightRenderParticleSampler::~SandHeightRenderParticleSampler()
	{
		m_normalizePosition.Release();
		//bExist.Release();
	}

	void SandHeightRenderParticleSampler::Initalize(int nx, int ny, int freq, int layer, float gridLength)
	{
		m_nx = nx;
		m_ny = ny;
		var_SampleFreq.setValue(freq);
		var_SampleLayer.setValue(layer);
		m_gridLength = gridLength;
		m_spacing = 1.0f / freq;

		m_normalizePosition.Resize(m_nx * freq, m_ny * freq, layer);
		this->Generate();
		//this->compute();
	}

	__global__ void g_generateParticles(Grid3f position)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		int nx = position.Nx();
		int ny = position.Ny();
		int layer = position.Nz();

		if (i >= nx) return;
		if (j >= ny) return;

		RandNumber gen(position.Index(i, j, 0));

		for (int k = 0; k < layer; ++k)
		{
			float x, y, z;
			x = gen.Generate();
			y = gen.Generate();
			z = gen.Generate();

			position(i, j, k) = make_float3(x, y, z);
		}
	}

	void SandHeightRenderParticleSampler::Generate()
	{
		dim3 bDims(16, 16, 1);
		dim3 gDims(cudaGridSize(m_nx * var_SampleFreq.getValue(), bDims.x), 
			cudaGridSize(m_ny * var_SampleFreq.getValue(), bDims.y), 1);

		g_generateParticles << <gDims, bDims >> > (m_normalizePosition);
		cuSynchronize();
	}

	


	__host__ __device__ vertex sampToVertex(float3 gp, float h, int width, int height, float dl)
	{
		vertex v;

		if (gp.x < 0 || gp.y < 0 || gp.z < 0)
		{
			v.x = 0;
			v.z = 0;
			v.y = 0;
		}
		else
		{
			v.x = (gp.x/* - Nx / 2*/)* dl;
			v.z = (gp.z/* - Ny / 2*/)* dl;
			v.y = -gp.y * dl + h;// +100 * dl;
		}

		//v = v * 0.05f;

		return v;
	}


	/**
	*@brief Sample sand surface particle for visualization.
	*@details Assumption: Height field plane is Ozx.
 	*/
	//__global__ void g_sampleSurfacePoints(Vector3f* samples, Grid3f position, 
	//	int width, int height, float sample_spacing, float grid_spacing)
	//{

	//	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//	int j = threadIdx.y + blockIdx.y * blockDim.y;

	//	if (i >= position.Nx() || j >= position.Ny())
	//		return;

	//	int layer = position.Nz();
	//	for (int k = 0; k < layer; k++)
	//	{
	//		float3 pos = position(i, j, k);
	//		float grid_fv = (i + pos.x)*sample_spacing;
	//		float grid_fu = (j + pos.y)*sample_spacing;

	//		if (grid_fv < 0.0f) grid_fv = 0.0f;
	//		if (grid_fv > width - 1) grid_fv = width - 1.0f;
	//		if (grid_fu < 0.0f) grid_fu = 0.0f;
	//		if (grid_fu > height - 1) grid_fu = height - 1.0f;

	//		int gridv = floor(grid_fv);		int gridu = floor(grid_fu);
	//		float fv = grid_fv - gridv;		float fu = grid_fu - gridu;

	//		if (gridv == width - 1) { gridv = width - 2; fv = 1.0f; }
	//		if (gridu == height - 1) { gridu = height - 2; fu = 1.0f; }

	//		float w00 = (1.0f - fv)*(1.0f - fu);
	//		float w10 = fv * (1.0f - fu);
	//		float w01 = (1.0f - fv)*fu;
	//		float w11 = fv * fu;

	//		float4 gp00 = tex2D(texture_grid, gridv, gridu);
	//		float4 gp10 = tex2D(texture_grid, gridv + 1, gridu);
	//		float4 gp01 = tex2D(texture_grid, gridv, gridu + 1);
	//		float4 gp11 = tex2D(texture_grid, gridv + 1, gridu + 1);

	//		float4 gp = w00 * gp00 + w10 * gp10 + w01 * gp01 + w11 * gp11;
	//		float4 samplep = sampToVertex(make_float3(grid_fv,  k*sample_spacing, grid_fu), 
	//			gp.x + gp.w, width, height, grid_spacing);
	//		samples[position.Index(i, j, k)] = Vector3f(samplep.x, samplep.y, samplep.z);
	//	}
	//}

	//void RenderParticleSampler::doSampling(Vector3f* pointSample, DeviceArrayPitch2D4f & sandGrid)
	//{
	//	SSEUtil::bindTexture2D(texture_grid, sandGrid);

	//	dim3 bDims(16, 16, 1);
	//	dim3 gDims(cudaGridSize(m_nx * m_sampleFreq, bDims.x), cudaGridSize(m_ny * m_sampleFreq, bDims.y), 1);
	//	g_sampleSurfacePoints << <gDims, bDims >> > (pointSample, m_normalizePosition, m_nx, m_ny, m_spacing, m_gridLength);
	//	
	//	cuSynchronize();


	//}


	__global__ void PFSandSampler_sampleSurfacePoints(Vector3f* samples, Grid3f position,
		DeviceHeightField1d sandHeight, DeviceHeightField1d landHeight, float sampleSpacing, float hThreshold=0.002)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i >= position.Nx() || j >= position.Ny())
			return;

		int layer = position.Nz();

		for (int k = 0; k < layer; k++)
		{
			float3 posLoc = position(i, j, k);
			float grid_fv = (i + posLoc.x)*sampleSpacing;
			float grid_fu = (j + posLoc.y)*sampleSpacing;

			Vector3d posCen = sandHeight.gridCenterPosition((int)grid_fv, (int)grid_fu);
			posCen[0] += (grid_fv - (int)grid_fv) * sandHeight.getDx();
			posCen[2] += (grid_fu - (int)grid_fu) * sandHeight.getDz();

			//if (i == 0 && j == 0)
			//	printf("%lf %lf, %lf %lf\n", posCen[0], posCen[2], sandHeight.getDx(), sandHeight.getDz());
			
			Vector3f pos;
			double curh = sandHeight.get(posCen[0], posCen[2]);
			if (abs(curh) > hThreshold)
			{
				pos[0] = posCen[0];	pos[2] = posCen[2];
				pos[1] = landHeight.get(posCen[0], posCen[2])
					+ sandHeight.get(posCen[0], posCen[2]);
			}

			samples[position.Index(i, j, k)] = pos;
		}
	}

	//void RenderParticleSampler::doSampling(Vector3f * pointSample, DeviceHeightField1d & sandHeight, DeviceHeightField1d & landHeight)
	//{
	//	uint3 gsize;
	//	int sampleFreq = var_SampleFreq.getValue();
	//	gsize.x = m_nx * sampleFreq;
	//	gsize.y = m_ny * sampleFreq;
	//	gsize.z = 1;

	//	cuExecute2D(gsize, ParSampler_sampleSurfacePoints,
	//		pointSample, m_normalizePosition,
	//		sandHeight, landHeight,
	//		m_spacing
	//	);

	//}

	void SandHeightRenderParticleSampler::compute()
	{


		if (!m_sandHeight || !m_landHeight)
			return;

		Node* pnode = this->getParent();
		if (!pnode)
			return;
		auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pnode->getTopologyModule());
		if (!pointset)
			return;

		

		auto& points = pointset->getPoints();

		uint3 gsize;
		int sampleFreq = var_SampleFreq.getValue();
		gsize.x = m_normalizePosition.Nx();
		gsize.y = m_normalizePosition.Ny();
		gsize.z = 1;

		if (points.size() != gsize.x*gsize.y* m_normalizePosition.Nz())
		{
			points.resize(gsize.x*gsize.y* m_normalizePosition.Nz());
		}

		

		cuExecute2D(gsize, PFSandSampler_sampleSurfacePoints,
			points.begin(), m_normalizePosition,
			*m_sandHeight, *m_landHeight,
			m_spacing
		);

		//HostArray<Vector3f> hostPoints;
		//hostPoints.resize(points.size());
		//Function1Pt::copy(hostPoints, points);

		//hostPoints.release();

		//cudaDeviceSynchronize();				
		//cudaError_t err = cudaGetLastError();	

	}



	void ParticleSandRenderSampler::Initialize(std::shared_ptr<PBDSandSolver> solver)
	{
		particleType = &(solver->getParticleTypes());
		particlePos = &(solver->getParticlePosition());
		particleRho2D = &(solver->getParticleRho2D());
		landHeight = &(solver->getLand());

		rho0 = solver->getRho0();

		//this->compute();
	}

	__global__ void ParSandSampler_sampleRenderPoints(
		DeviceArray<Vector3f> point,
		DeviceHeightField1d land,
		DeviceDArray<Vector3d> parPos,
		DeviceDArray<double> parRho2D,
		DeviceDArray<ParticleType> parType,
		double rho0
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= parPos.size())
			return;

		Vector3d pos;
		if (parType[tid] == ParticleType::SAND)
		{
			pos = parPos[tid];
			pos[1] = parRho2D[tid] / rho0 + land.get(pos[0], pos[2]);
		}
		point[tid] = Vector3f(pos[0], pos[1], pos[2]);
	}

	void ParticleSandRenderSampler::compute()
	{
		if (!landHeight || !particlePos || !particleRho2D || !particleType)
			return;
		if (particlePos->size() <= 0 || particleRho2D->size() <= 0)
			return;


		Node* pnode = this->getParent();
		if (!pnode)
			return;
		auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pnode->getTopologyModule());
		if (!pointset)
			return;
		auto& points = pointset->getPoints();
		if (points.size() != particlePos->size())
			points.resize(particlePos->size());

		cuExecute(particlePos->size(), ParSandSampler_sampleRenderPoints,
			points,
			*landHeight,
			*particlePos, *particleRho2D, *particleType,
			rho0
		);

		//HostArray<Vector3f> hostpos;
		//hostpos.resize(points.size());
		//Function1Pt::copy(hostpos, points);

		//HostDArray<double> hostrho;
		//hostrho.resize(particleRho2D->size());
		//Function1Pt::copy(hostrho, *particleRho2D);

		//HostDArray<ParticleType> hosttype;
		//hosttype.resize(particleType->size());
		//Function1Pt::copy(hosttype, *particleType);


		//Vector3f totalPos = thrust::reduce(thrust::device, points.begin(), points.begin() + points.size(), Vector3f(),
		//	thrust::plus<Vector3f>());

		//int parnum = thrust::count(thrust::device, particleType->begin(), particleType->begin() + particleType->size(),
		//	ParticleType::SAND);
		//totalPos /= parnum;

		//hosttype.release();
		//hostrho.release();
		//hostpos.release();
	}

}




