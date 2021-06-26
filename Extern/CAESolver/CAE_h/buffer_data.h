#pragma once
#include<vector>
#include<iostream>
#include<string>
#include <cuda_runtime.h>
#include"helper_cuda.h"

template<typename T>
struct BufferData
{
	/**
	数据列
	*/
	std::vector<T> dataArray_;

	/**
	cpu数据指针
	*/
	T* ptrCpu_;

	/**
	gpu数据指针列，用于存放各个GPU上的数据指针
	*/
	std::vector<T*> ptrGpuArray_;

	/**
	辅gpu存放在主gpu上的中间变量
	*/
	std::vector<T*> ptrMediumGpuArray_;

	/**
	数据长度
	*/
	int size_;

	/**
	算力大于6.0的GPU的编号列
	*/
	static std::vector<int> gpuIdArray_;

	/**
	各个GPU上的流，静态成员，该结构体共有
	*/
	static std::vector<cudaStream_t> streamArray_;

	/**
	各个辅gpu上与主gpu刚享的部分id存储在Gpu上的
	*/
	static std::vector<int*> sharedIdStoreGpu_;

	/**
	各个辅gpu上与主gpu刚享的部分id存储在Cpu上的
	*/
	static std::vector<int*> sharedIdStoreCpu_;

	/**
	各个辅gpu上与主gpu刚享的部分id总数
	*/
	static std::vector<int> sharedSize_;

	/**
	gpu数量，静态成员，该结构体共有
	*/
	static int gpuNum;

	/**
	将gpu数据拷贝到cpu中
	*/
	void gpuToCpu(const int gpuId = 0)
	{
		cudaSetDevice(gpuId);
		if (ptrGpuArray_[gpuId] == nullptr)
		{
			std::cout << "ERROR: can't copy null data from GPU " << gpuId << "to CPU " << __FILE__ << __LINE__ << std::endl;
		}
		if (ptrCpu_ == nullptr)
		{
			allocateCpu();
		}
		checkCudaErrors(cudaMemcpyAsync(ptrGpuArray_[gpuId], ptrCpu_, size_ * sizeof(T), cudaMemcpyDeviceToHost,streamArray_[gpuId]));
	}

	/**
	将cpu数据拷贝到所有gpu中
	*/
	void cpuToAllGpu()
	{
		if (ptrCpu_ == nullptr)
		{
			std::cout << "ERROR: cpu ptr is nullptr" << __FILE__ << __LINE__ << std::endl;
		}
		for (int in = 0; in < gpuNum; in++)
		{
			cudaSetDevice(gpuIdArray_[in]);
			checkCudaErrors(cudaMemcpyAsync(ptrCpu_, ptrGpuArray_[in], size_ * sizeof(T), cudaMemcpyHostToDevice, streamArray_[in]));
		}
	}

	/**
	将cpu数据拷贝到特定gpu中
	*/
	void cpuToAssignGpu(const int gpuId)
	{
		cudaSetDevice(gpuIdArray_[gpuId]);
		if (ptrCpu_ == nullptr)
		{
			std::cout << "ERROR: cpu ptr is nullptr" << __FILE__ << __LINE__ << std::endl;
		}
		if (ptrGpuArray_[gpuId] == nullptr)
		{
			checkCudaErrors(cudaMalloc(&ptrGpuArray_[gpuId], size_ * sizeof(T)));
		}
		checkCudaErrors(cudaMemcpyAsync(ptrCpu_, ptrGpuArray_[gpuId], size_ * sizeof(T), cudaMemcpyHostToDevice, streamArray_[gpuId]));
	}

	/**
	将主gpu的数据复制到辅gpu中
	*/
	void mianGpuToAidGpu()
	{
		for (int in = 1; in < gpuNum; in++)
		{
			cudaSetDevice(gpuIdArray_[in]);
			checkCudaErrors(cudaMemcpyAsync(ptrGpuArray_[0], 
				ptrGpuArray_[in], size_ * sizeof(T), cudaMemcpyDeviceToDevice, streamArray_[in]));
		}
	}

	/**
	将辅gpu的数据复制到主Gpu中
	*/
	void aidGpuToMainGpu();

	/**
	释放所有gpu数据
	*/
	void releaseAllGpu()
	{
		for (int in = 0; in < gpuNum; in++)
		{
			checkCudaErrors(cudaFree(ptrGpuArray_[in]));
			ptrGpuArray_[in] = nullptr;
		}
	}

	/**
	释放所有辅助gpu的中间变量
	*/
	void releaseAllMediumGpu()
	{
		for (int in = 0; in < gpuNum - 1; in++)
		{
			checkCudaErrors(cudaFree(ptrMediumGpuArray_[in]));
			ptrMediumGpuArray_[in] = nullptr;
		}
	}

	/**
	释放cpu数据
	*/
	void releaseCpu()
	{
		if (ptrCpu_ != nullptr)
			dataArray_.clear();
		ptrCpu_ = nullptr;
	}

	/**
	cpu数据置零
	*/
	void setZeroCpu()
	{
		if (ptrCpu_ != nullptr)
		{
			memset(ptrCpu_, 0, size_ * sizeof(T));
		}
	}

	/**
	所有Gpu数据置零
	*/
	void setZeroAllGpu()
	{
		for (int in = 0; in < gpuNum; in++)
		{
			cudaSetDevice(gpuIdArray_[in]);
			checkCudaErrors(cudaMemsetAsync(ptrGpuArray_[in], 0, size_ * sizeof(T), streamArray_[in]));
		}
	}

	/**
	分配cpu数据
	*/
	void allocateCpu()
	{
		if (ptrCpu_ == nullptr&&size_ > 0)
		{
			dataArray_.resize(size_);
			ptrCpu_ = dataArray_.data();
		}
	}

	/**
	分配gpu数据
	*/
	void allocateAllGpu()
	{
		ptrGpuArray_.resize(gpuNum);
		for (int in = 0; in < gpuNum; in++)
		{
			cudaSetDevice(gpuIdArray_[in]);
			checkCudaErrors(cudaMalloc(&ptrGpuArray_[in], size_ * sizeof(T)));
		}
	}

	/**
	分配主gpu上的中间变量数据
	*/
	void allocateMediumPtrGpu()
	{
		ptrMediumGpuArray_.resize(gpuNum);
		ptrMediumGpuArray_[0] = nullptr;
		for (int in = 1; in < gpuNum; in++)
		{
			cudaSetDevice(0);
			checkCudaErrors(cudaMalloc(&ptrMediumGpuArray_[in],sharedSize_[in]*sizeof(T)));
		}
	}
};
