#include "Reduction.h"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include "cuda_utilities.h"
#include "sharedmem.h"
#include "Functional.h"

namespace Physika {

	template<typename T>
	Reduction<T>::Reduction(unsigned num)
		: m_num(num)
		, m_aux(NULL)
	{
		m_auxNum = GetAuxiliaryArraySize(num);
		cudaMalloc((void**)&m_aux, m_auxNum * sizeof(T));
	}

	template<typename T>
	Reduction<T>::~Reduction()
	{
		cudaFree(m_aux);
	}

	template<typename T>
	Reduction<T>* Reduction<T>::Create(int n)
	{
		return new Reduction<T>(n);
	}

	/*!
	*	\brief	Reduction using maximum of float values in shared memory for a warp.
	*/
	template <typename T, 
			  unsigned blockSize,
			  typename Function>
	__device__ 	void KerReduceWarp(volatile T* pData, unsigned tid, Function func)
	{
		if (blockSize >= 64)pData[tid] = func(pData[tid], pData[tid + 32]);
		if (blockSize >= 32)pData[tid] = func(pData[tid], pData[tid + 16]);
		if (blockSize >= 16)pData[tid] = func(pData[tid], pData[tid + 8]);
		if (blockSize >= 8)pData[tid] = func(pData[tid], pData[tid + 4]);
		if (blockSize >= 4)pData[tid] = func(pData[tid], pData[tid + 2]);
		if (blockSize >= 2)pData[tid] = func(pData[tid], pData[tid + 1]);
	}

	/*!
	*	\brief	Accumulates the sum of n values of array pData[], 
	*	storing the result in the beginning of res[].
	*	(Many positions of res[] are used as blocks, storing the final result in res[0]).
	*/
	template <typename T, 
			  unsigned blockSize,
			  typename Function>
	__global__ void KerReduce(const T *pData, unsigned n, T *pAux, Function func, T val)
	{
		//extern __shared__ T sharedMem[];

		SharedMemory<T> smem;
		T* sharedMem = smem.getPointer();

		unsigned tid = threadIdx.x;
		unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
		sharedMem[tid] = (id < n ? pData[id] : val);
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 64]);   __syncthreads(); }
		if (tid < 32)KerReduceWarp<T, blockSize>(sharedMem, tid, func);
		if (tid == 0)pAux[blockIdx.x] = sharedMem[0];
	}

	template<typename T, typename Function>
	T Reduce(T* pData, unsigned num, T* pAux, Function func, T v0)
	{
		unsigned n = num;
		unsigned sharedMemSize = REDUCTION_BLOCK * sizeof(T);
		unsigned blockNum = cudaGridSize(num, REDUCTION_BLOCK);
		T* subData = pData;
		T* aux1 = pAux;
		T* aux2 = pAux + blockNum;
		T* subAux = aux1;
		while (n > 1) {
			KerReduce<T, REDUCTION_BLOCK, Function> << <blockNum, REDUCTION_BLOCK, sharedMemSize >> > (subData, n, subAux, func, v0);
			n = blockNum; 
			blockNum = cudaGridSize(n, REDUCTION_BLOCK);
			if (n > 1) {
				subData = subAux; subAux = (subData == aux1 ? aux2 : aux1);
			}
		}

		T val;
		if (num > 1)
			cudaMemcpy(&val, subAux, sizeof(T), cudaMemcpyDeviceToHost);
		else 
			cudaMemcpy(&val, pData, sizeof(T), cudaMemcpyDeviceToHost);

		return val;
	}

	template<typename T>
	T Physika::Reduction<T>::Accumulate(T* val, int num)
	{
		assert(num == m_num);
		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0);
	}

	template<typename T>
	T Physika::Reduction<T>::Maximum(T* val, int num)
	{
		assert(num == m_num);
		return Reduce(val, num, m_aux, MaximumFunc<T>(), (T)-FLT_MAX);
	}

	template<typename T>
	T Physika::Reduction<T>::Minimum(T* val, int num)
	{
		assert(num == m_num);
		return Reduce(val, num, m_aux, MinimumFunc<T>(), (T)FLT_MAX);
	}

	template<typename T>
	T Physika::Reduction<T>::Average(T* val, int num)
	{
		assert(num == m_num);
		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0) / num;
	}
}