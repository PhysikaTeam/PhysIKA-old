#include <cuda_runtime.h>
#include <vector_types.h>
#include "Core/Utility.h"

namespace PhysIKA
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
		void plus(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.size(), PlusFunc<T>());

		}

		template <typename T>
		void subtract(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc <<<pDim, BLOCK_SIZE >>> (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.size(), MinusFunc<T>());
		}


		template <typename T>
		void multiply(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.size(), MultiplyFunc<T>());

		}

		template <typename T>
		void divide(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), zArr.size(), DivideFunc<T>());

		}


		template <typename T>
		void saxpy(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr, T alpha)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerSaxpy << <pDim, BLOCK_SIZE >> > (zArr.getDataPtr(), xArr.getDataPtr(), yArr.getDataPtr(), alpha, zArr.size());
		}

		template void plus(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void plus(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void plus(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void subtract(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void subtract(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void subtract(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void multiply(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void multiply(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void multiply(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void divide(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void divide(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void divide(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void saxpy(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&, float);
		template void saxpy(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&, double);
	}
}