#pragma once

#include <assert.h>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include "Platform.h"

namespace Physika {

	/** check whether cuda thinks there was an error and fail with msg, if this is the case
	* @ingroup tools
	*/
	static inline void checkCudaError(const char *msg) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
		}
	}

	// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#  define cuvSafeCall(X)  \
      if(strcmp(#X,"cudaThreadSynchronize()")!=0){ X; Physika::checkCudaError(#X); }
#else
#  define cuvSafeCall(X) X; Physika::checkCudaError(#X);
#endif

// 	template<typename T>
// 	void CopyMemory(T* dst, T*src, size_t size, DeviceType dstType, DeviceType srcType)
// 	{
// 
// 	}


	template<DeviceType deviceType>
	class MemoryManager {

	public:

		virtual ~MemoryManager() {
		}

		virtual void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) = 0;

		virtual void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) = 0;

		virtual void initMemory(void* ptr, int value, size_t count) = 0;

		virtual void releaseMemory(void** ptr) = 0;
	};

	/**
	 * Allocator allows allocation, deallocation and copying depending on memory_space_type
	 *
	 * \ingroup tools
	 */
	template<DeviceType deviceType>
	class DefaultMemoryManager : public MemoryManager<deviceType> {

	public:

		virtual ~DefaultMemoryManager() {
		}

		void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) override;

		void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) override;

		void initMemory(void* ptr, int value, size_t count) override;

		void releaseMemory(void** ptr) override;

	};


	/**
	 * @brief allocator that uses cudaMallocHost for allocations in host_memory_space
	 */
	template<DeviceType deviceType>
	class CudaMemoryManager : public DefaultMemoryManager<deviceType> {

	public:

		virtual ~CudaMemoryManager() {
		}

		void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) override;

		void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) override;

		void initMemory(void* ptr, int value, size_t count) override;

		void releaseMemory(void** ptr) override;

	};



	template class DefaultMemoryManager<DeviceType::CPU>;
	template class DefaultMemoryManager<DeviceType::GPU>;
	template class CudaMemoryManager<DeviceType::CPU>;
	template class CudaMemoryManager<DeviceType::GPU>;
}