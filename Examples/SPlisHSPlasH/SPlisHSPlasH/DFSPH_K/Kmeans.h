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
		//ÿ���߳̿����̵߳�����
		unsigned int TPB;
		unsigned int sharedMemorySize;

		//�ظ���
		unsigned int m_k;
		//�����ģ�[0]:�����������ģ�[1]:�����������ģ�
		//��һ�־��������
		std::vector<Real> m_clusterCentroids;
		//��һ�־��������
		std::vector<Real> m_new_clusterCentroids;
		// particles cluster v max
		std::vector<Real> m_clusterVMax;

		//�豸�ٶ���������ָ��
		float* dev_v;
		//�豸�ٶȱ�������ָ��
		float* dev_vr;
		//�豸���ӷ�������ָ��
		unsigned int* dev_cluster;
		//�豸����������ָ��
		float* dev_clusterCentroids;
		//�豸������ٶȱ�������ָ��
		float* dev_clusterVMax;

		//���ڹ�Լ
		//���ڴ�0���ٶȡ�����������ٶȣ����ڴ�1���ٶȡ�����������ٶ�
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