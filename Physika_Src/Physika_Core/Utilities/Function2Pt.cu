#include <cuda_runtime.h>
#include <vector_types.h>
#include "Physika_Core/Utilities/Function2Pt.h"
#include "Physika_Core/Utilities/Functional.h"
#include "Physika_Core/Utilities/cuda_utilities.h"

namespace Physika
{
	namespace Function2Pt
	{
		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a1, T* a2, int num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(a1[pId], a2[pId]);
		}

		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a2, int num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(out[pId], a2[pId]);
		}

		template <typename T>
		__global__ void KerSaxpy(T *zArr, T* xArr, T* yArr, T alpha, int num)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			zArr[pId] = alpha * xArr[pId] + yArr[pId];
		}


		template <typename T>
		void Plus(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.Size() == xArr.Size() && zArr.Size() == yArr.Size());
			unsigned pDim = cudaGridSize(zArr.Size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.Size(), PlusFunc<T>());

		}

		template <typename T>
		void Subtract(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.Size() == xArr.Size() && zArr.Size() == yArr.Size());
			unsigned pDim = cudaGridSize(zArr.Size(), BLOCK_SIZE);
			KerTwoPointFunc <<<pDim, BLOCK_SIZE >>> (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.Size(), MinusFunc<T>());
		}


		template <typename T>
		void Multiply(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.Size() == xArr.Size() && zArr.Size() == yArr.Size());
			unsigned pDim = cudaGridSize(zArr.Size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.Size(), MultiplyFunc<T>());

		}

		template <typename T>
		void Divide(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.Size() == xArr.Size() && zArr.Size() == yArr.Size());
			unsigned pDim = cudaGridSize(zArr.Size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.Size(), DivideFunc<T>());

		}


		template <typename T>
		void Saxpy(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr, T alpha)
		{
			assert(zArr.Size() == xArr.Size() && zArr.Size() == yArr.Size());
			unsigned pDim = cudaGridSize(zArr.Size(), BLOCK_SIZE);
			KerSaxpy << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), alpha, zArr.Size());
		}
	}
}