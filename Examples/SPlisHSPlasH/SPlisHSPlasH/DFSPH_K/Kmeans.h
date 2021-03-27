#ifndef __Kmeans_h__
#define __Kmeans_h__

#include <vector>
#include "SPlisHSPlasH/Common.h"
#include "SimulationDataDFSPH_K.h"
#include "cuda_runtime.h"

namespace SPH
{
	class Kmeans
	{
	private:
		//每个线程块中线程的数量
		unsigned int TPB;
		unsigned int sharedMemorySize;

		//簇个数
		unsigned int m_k;
		//簇中心（[0]:快速粒子中心；[1]:慢速粒子中心）
		//上一轮聚类簇中心
		std::vector<Real> m_clusterCentroids;
		//新一轮聚类簇中心
		std::vector<Real> m_new_clusterCentroids;
		// particles cluster v max
		std::vector<Real> m_clusterVMax;

		//设备速度向量数组指针
		float* dev_v;
		//设备速度标量数组指针
		float* dev_vr;
		//设备粒子分类数组指针
		unsigned int* dev_cluster;
		//设备簇中心数组指针
		float* dev_clusterCentroids;
		//设备簇最大速度标量数组指针
		float* dev_clusterVMax;

		//用于归约
		//属于簇0的速度、计数、最大速度；属于簇1的速度、计数、最大速度
		float* dev_cluster0_v;
		float* dev_cluster1_v;
		int* dev_cluster0_count;
		int* dev_cluster1_count;
		float* dev_cluster0_vmax;
		float* dev_cluster1_vmax;

	public:
		Kmeans();
		~Kmeans();
		void init();
		void performKmeans(SimulationDataDFSPH_K& simulationData);

		FORCE_INLINE const Real getClusterVMax(const unsigned int i) const
		{
			return m_clusterVMax[i];
		}

		FORCE_INLINE Real& getClusterVMax(const unsigned int i)
		{
			return m_clusterVMax[i];
		}

		FORCE_INLINE void setClusterVMax(const unsigned int i, const Real v)
		{
			m_clusterVMax[i] = v;
		}
	};
}


#endif