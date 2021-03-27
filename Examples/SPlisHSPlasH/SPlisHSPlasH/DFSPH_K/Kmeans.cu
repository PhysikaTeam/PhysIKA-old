#include "Kmeans.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/DFSPH_K/SimulationDataDFSPH_K.h"

using namespace SPH;

inline __device__ float squaredDistance(const float v, const float vc) {
	return (v - vc) * (v - vc);
}

__global__ void vector2Real(float* v, float* vr, int numAllActiveParticles) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= numAllActiveParticles)
		return;
	const float vx = v[tid * 3];
	const float vy = v[tid * 3 + 1];
	const float vz = v[tid * 3 + 2];
	vr[tid] = sqrtf(vx * vx + vy * vy + vz * vz);
}

__global__ void kMeansClusterAssignment(
	float* vel,
	unsigned int* cluster,
	int numAllActiveParticles,
	float* clusterCentroids,
	unsigned int k,
	float* cluster0_v,
	float* cluster1_v,
	int* cluster0_count,
	int* cluster1_count,
	float* cluster0_vmax,
	float* cluster1_vmax
) {
	extern __shared__ float shared_data[];
	const int local_tid = threadIdx.x;
	const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;


	if (global_tid >= numAllActiveParticles)
		return;
	if (local_tid < k) {
		shared_data[local_tid] = clusterCentroids[local_tid];
	}
	__syncthreads();

	const float v = vel[global_tid];
	float best_distance = FLT_MAX;
	int best_cluster = -1;
	for (int i = 0; i < k; ++i) {
		const float distance = squaredDistance(v, shared_data[i]);
		if (distance < best_distance) {
			best_distance = distance;
			best_cluster = i;
		}
	}
	cluster[global_tid] = best_cluster;
	__syncthreads();

	const int x = local_tid;
	const int y = local_tid + blockDim.x;
	const int z = local_tid + blockDim.x * 2;
	for (int i = 0; i < k; ++i) {
		shared_data[x] = (best_cluster == i) ? v : 0;
		shared_data[y] = (best_cluster == i) ? 1 : 0;
		shared_data[z] = (best_cluster == i) ? v : 0;
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
			if (x < stride) {
				shared_data[x] += shared_data[x + stride];
				shared_data[y] += shared_data[y + stride];
				shared_data[z] = (shared_data[z] > shared_data[z + stride]) ? shared_data[z] : shared_data[z + stride];
			}
			__syncthreads();
		}

		if (x == 0) {
			switch (i)
			{
			case 0:
				cluster0_v[blockIdx.x] = shared_data[x];
				cluster0_count[blockIdx.x] = shared_data[y];
				cluster0_vmax[blockIdx.x] = shared_data[z];
				break;
			case 1:
				cluster1_v[blockIdx.x] = shared_data[x];
				cluster1_count[blockIdx.x] = shared_data[y];
				cluster1_vmax[blockIdx.x] = shared_data[z];
				break;
			default:
				break;
			}
			__syncthreads();
		}
	}
}

__global__ void secondReduction(
	float* cluster0_v,
	float* cluster1_v,
	int* cluster0_count,
	int* cluster1_count,
	float* cluster0_vmax,
	float* cluster1_vmax
) {
	extern __shared__ float shared_data[];
	const int local_tid = threadIdx.x;
	const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	const int x = local_tid;
	const int y = local_tid + blockDim.x;
	const int z = local_tid + blockDim.x * 2;

	shared_data[x] = cluster0_v[global_tid];
	cluster0_v[global_tid] = 0;
	shared_data[y] = cluster0_count[global_tid];
	cluster0_count[global_tid] = 0;
	shared_data[z] = cluster0_vmax[global_tid];
	cluster0_vmax[global_tid] = 0;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (x < stride) {
			shared_data[x] += shared_data[x + stride];
			shared_data[y] += shared_data[y + stride];
			shared_data[z] = (shared_data[z] > shared_data[z + stride]) ? shared_data[z] : shared_data[z + stride];
		}
		__syncthreads();
	}

	if (x == 0) {
		cluster0_v[blockIdx.x] = shared_data[x];
		cluster0_count[blockIdx.x] = shared_data[y];
		cluster0_vmax[blockIdx.x] = shared_data[z];
	}
	__syncthreads();

	shared_data[x] = cluster1_v[global_tid];
	cluster1_v[global_tid] = 0;
	shared_data[y] = cluster1_count[global_tid];
	cluster1_count[global_tid] = 0;
	shared_data[z] = cluster1_vmax[global_tid];
	cluster1_vmax[global_tid] = 0;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (x < stride) {
			shared_data[x] += shared_data[x + stride];
			shared_data[y] += shared_data[y + stride];
			shared_data[z] = (shared_data[z] > shared_data[z + stride]) ? shared_data[z] : shared_data[z + stride];
		}
		__syncthreads();
	}

	if (x == 0) {
		cluster1_v[blockIdx.x] = shared_data[x];
		cluster1_count[blockIdx.x] = shared_data[y];
		cluster1_vmax[blockIdx.x] = shared_data[z];
	}
	__syncthreads();
}

__global__ void kMeansCentroidUpdate(
	float* clusterCentroids,
	float* clusterVMax,
	float* cluster0_v,
	float* cluster1_v,
	int* cluster0_count,
	int* cluster1_count,
	float* cluster0_vmax,
	float* cluster1_vmax
) {
	extern __shared__ float shared_data[];

	const int x = threadIdx.x;
	const int y = threadIdx.x + blockDim.x;
	const int z = threadIdx.x + blockDim.x * 2;

	shared_data[x] = cluster0_v[x];
	cluster0_v[x] = 0;
	shared_data[y] = cluster0_count[x];
	cluster0_count[x] = 0;
	shared_data[z] = cluster0_vmax[x];
	cluster0_vmax[x] = 0;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (x < stride) {
			shared_data[x] += shared_data[x + stride];
			shared_data[y] += shared_data[y + stride];
			shared_data[z] = (shared_data[z] > shared_data[z + stride]) ? shared_data[z] : shared_data[z + stride];
		}
		__syncthreads();
	}

	if (x == 0) {
		const int count = (shared_data[y] >= 1) ? shared_data[y] : 1;
		clusterCentroids[0] = shared_data[x] / count;
		clusterVMax[0] = shared_data[z];
	}
	__syncthreads();

	shared_data[x] = cluster1_v[x];
	cluster1_v[x] = 0;
	shared_data[y] = cluster1_count[x];
	cluster1_count = 0;
	shared_data[z] = cluster1_vmax[x];
	cluster1_vmax[x] = 0;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (x < stride) {
			shared_data[x] += shared_data[x + stride];
			shared_data[y] += shared_data[y + stride];
			shared_data[z] = (shared_data[z] > shared_data[z + stride]) ? shared_data[z] : shared_data[z + stride];
		}
		__syncthreads();
	}

	if (x == 0) {
		const int count = (shared_data[y] >= 1) ? shared_data[y] : 1;
		clusterCentroids[1] = shared_data[x] / count;
		clusterVMax[1] = shared_data[z];
	}
	__syncthreads();
}

Kmeans::Kmeans() :
	m_clusterCentroids(),
	m_new_clusterCentroids(),
	m_clusterVMax()
{
	TPB = 1024;
	sharedMemorySize = TPB * 3 * sizeof(float);

	m_k = 2;

	dev_v = nullptr;
	dev_vr = nullptr;
	dev_cluster = nullptr;
	dev_clusterCentroids = nullptr;
	dev_clusterVMax = nullptr;

	dev_cluster0_v = nullptr;
	dev_cluster1_v = nullptr;
	dev_cluster0_count = nullptr;
	dev_cluster1_count = nullptr;
	dev_cluster0_vmax = nullptr;
	dev_cluster1_vmax = nullptr;
}
Kmeans::~Kmeans()
{
	cudaFree(dev_v);
	cudaFree(dev_vr);
	cudaFree(dev_cluster);
	cudaFree(dev_clusterCentroids);
	cudaFree(dev_clusterVMax);
	cudaFree(dev_cluster0_v);
	cudaFree(dev_cluster1_v);
	cudaFree(dev_cluster0_count);
	cudaFree(dev_cluster1_count);
	cudaFree(dev_cluster0_vmax);
	cudaFree(dev_cluster1_vmax);
}
void Kmeans::init()
{
	cudaError_t cudaStatus;

	Simulation* sim = Simulation::getCurrent();
	const unsigned int numFluidModel = sim->numberOfFluidModels();
	unsigned int numAllParticles = 0;

	for (unsigned int i = 0; i < numFluidModel; i++)
	{
		FluidModel* fm = sim->getFluidModel(i);
		numAllParticles += fm->numParticles();
	}
	unsigned int max_Blocks1 = (numAllParticles + TPB - 1) / TPB;

	m_clusterCentroids.resize(m_k);
	m_new_clusterCentroids.resize(m_k);
	m_clusterVMax.resize(m_k);

	cudaStatus = cudaMalloc(&dev_v, numAllParticles * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaMalloc dev_v failed!");

	cudaStatus = cudaMalloc(&dev_vr, numAllParticles * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_vr failed!");

	cudaStatus = cudaMalloc(&dev_cluster, numAllParticles * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_cluster failed!");

	cudaStatus = cudaMalloc(&dev_clusterCentroids, m_k * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_clusterCentroids failed!");

	cudaStatus = cudaMalloc(&dev_clusterVMax, m_k * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_clusterVMax failed!");

	cudaStatus = cudaMalloc(&dev_cluster0_v, max_Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster0_v failed!");

	cudaStatus = cudaMalloc(&dev_cluster1_v, max_Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster1_v failed!");

	cudaStatus = cudaMalloc(&dev_cluster0_count, max_Blocks1 * TPB * sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster0_count failed!");

	cudaStatus = cudaMalloc(&dev_cluster1_count, max_Blocks1 * TPB * sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster1_count failed!");

	cudaStatus = cudaMalloc(&dev_cluster0_vmax, max_Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster0_vmax failed!");

	cudaStatus = cudaMalloc(&dev_cluster1_vmax, max_Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc cluster1_vmax failed!");

}

void Kmeans::performKmeans(SimulationDataDFSPH_K& simulationData)
{
	cudaError_t cudaStatus;

	Simulation* sim = Simulation::getCurrent();
	const unsigned int numFluidModel = sim->numberOfFluidModels();
	unsigned int numAllActiveParticles = 0;

	for (unsigned int i = 0; i < numFluidModel; i++)
	{
		FluidModel* fm = sim->getFluidModel(i);
		numAllActiveParticles += fm->numActiveParticles();
	}

	unsigned int Blocks1 = (numAllActiveParticles + TPB - 1) / TPB;
	unsigned int Blocks2 = (Blocks1 + TPB - 1) / TPB;

	unsigned int copyOffset = 0;
	bool iteration = true;

	//¿½±´ËÙ¶È
	for (unsigned int i = 0; i < numFluidModel; i++) {
		FluidModel* fm = sim->getFluidModel(i);
		const unsigned int numActiveParticles = fm->numActiveParticles();
		cudaStatus = cudaMemcpy(dev_v + copyOffset, &fm->getVelocity(0)[0], numActiveParticles * 3 * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMemcpy dev_v %d failed!", i);
		copyOffset += numActiveParticles * 3;
	}

	m_clusterCentroids[0] = sim->getMaxV();
	m_clusterCentroids[1] = sim->getThresholdV();
	cudaStatus = cudaMemcpy(dev_clusterCentroids, &m_clusterCentroids[0], m_k * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy dev_clusterCentroids failed!");

	cudaStatus = cudaMemset(dev_cluster0_v, 0, Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster0_v failed!");

	cudaStatus = cudaMemset(dev_cluster1_v, 0, Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster1_v failed!");

	cudaStatus = cudaMemset(dev_cluster0_count, 0, Blocks1 * TPB * sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster0_count failed!");

	cudaStatus = cudaMemset(dev_cluster1_count, 0, Blocks1 * TPB * sizeof(int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster1_count failed!");

	cudaStatus = cudaMemset(dev_cluster0_vmax, 0, Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster0_vmax failed!");

	cudaStatus = cudaMemset(dev_cluster1_vmax, 0, Blocks1 * TPB * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset cluster1_vmax failed!");

	vector2Real << <Blocks1, TPB >> > (dev_v, dev_vr, numAllActiveParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize 0 failed!");

	while (iteration) {

		kMeansClusterAssignment << <Blocks1, TPB, sharedMemorySize >> > (dev_vr, dev_cluster, numAllActiveParticles, dev_clusterCentroids, m_k,
			dev_cluster0_v, dev_cluster1_v, dev_cluster0_count, dev_cluster1_count, dev_cluster0_vmax, dev_cluster1_vmax);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize 1 failed!");

		if (Blocks2 > 1) {
			secondReduction << <Blocks2, TPB, sharedMemorySize >> > (dev_cluster0_v, dev_cluster1_v, dev_cluster0_count, dev_cluster1_count, dev_cluster0_vmax, dev_cluster1_vmax);
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize 2 failed!");
		}

		kMeansCentroidUpdate << <1, TPB, sharedMemorySize >> > (dev_clusterCentroids, dev_clusterVMax, dev_cluster0_v, dev_cluster1_v, dev_cluster0_count, dev_cluster1_count, dev_cluster0_vmax, dev_cluster1_vmax);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize 3 failed!");

		cudaStatus = cudaMemcpy(&m_new_clusterCentroids[0], dev_clusterCentroids, m_k * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMemcpy clusterCentroids failed!");

		iteration = false;
		for (int i = 0; i < m_k; ++i) {
			const float diff = abs(m_new_clusterCentroids[i] - m_clusterCentroids[i]);
			if (diff > 0.1) {
				iteration = true;
			}
			//std::cout << m_new_clusterCentroids[i] << " " << m_clusterCentroids[i] << " ";
			m_clusterCentroids[i] = m_new_clusterCentroids[i];
		}
		//std::cout << std::endl;
	}

	copyOffset = 0;
	for (unsigned int i = 0; i < numFluidModel; i++) {
		FluidModel* fm = sim->getFluidModel(i);
		const unsigned int numActiveParticles = fm->numActiveParticles();
		cudaStatus = cudaMemcpy(&simulationData.getCluster(i,0), dev_cluster + copyOffset, numActiveParticles * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMemcpy cluster %d failed!", i);
		copyOffset += numActiveParticles;
	}

	cudaStatus = cudaMemcpy(&getClusterVMax(0), dev_clusterVMax, m_k * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaMemcpy clusterVMax failed!");
	//std::cout << m_clusterVMax[0] << " " << m_clusterVMax[1] << std::endl;

}
