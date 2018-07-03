#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/Array2D.h"
#include "Physika_Core/Cuda_Array/Array3D.h"
/*
*  This file implements all one-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/
namespace Physika
{
	namespace Function1Pt
	{ 
		template<typename T, DeviceType dType1, DeviceType dType2>
		void Copy(Array<T, dType1>& arr1, Array<T, dType2>& arr2)
		{
			assert(arr1.Size() == arr2.Size());
			int totalNum = arr1.Size();
			if (arr1.isGPU() && arr2.isGPU())	(cudaMemcpy(arr1.getDataPtr(), arr2.getDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (arr1.isCPU() && arr2.isGPU())	(cudaMemcpy(arr1.getDataPtr(), arr2.getDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (arr1.isGPU() && arr2.isCPU())	(cudaMemcpy(arr1.getDataPtr(), arr2.getDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (arr1.isCPU() && arr2.isCPU())	memcpy(arr1.getDataPtr(), arr2.getDataPtr(), totalNum * sizeof(T));
		}

		template<typename T, DeviceType deviceType>
		void Copy(Array<T, deviceType>& arr, std::vector<T>& vec)
		{
			assert(vec.size() == arr.Size());
			int totalNum = arr.Size();
			switch (deviceType)
			{
			case CPU:
				memcpy(arr.getDataPtr(), &vec[0], totalNum * sizeof(T));
				break;
			case GPU:
				(cudaMemcpy(arr.getDataPtr(), &vec[0], totalNum * sizeof(T), cudaMemcpyHostToDevice));
				break;
			default:
				break;
			}
		}

		template<typename T, DeviceType dType1, DeviceType dType2>
		void Copy(Array2D<T, dType1>& g1, Array2D<T, dType1>& g2)
		{
			assert(g1.Size() == g2.Size() && g1.Nx()() == g2.Nx() && g2.Ny() == g2.Ny());
			int totalNum = g1.Size();
			if (g1.IsGPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.IsCPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.IsGPU() && g2.IsCPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.IsCPU() && g2.IsCPU())	memcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T));
		}

		template<typename T, DeviceType dType1, DeviceType dType2>
		void Copy(Array3D<T, dType1>& g1, Array3D<T, dType1>& g2)
		{
			assert(g1.Size() == g2.Size() && g1.Nx()() == g2.Nx() && g2.Ny() == g2.Ny() && g1.Nz() == g2.Nz());
			int totalNum = g1.Size();
			if (g1.IsGPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.IsCPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.IsGPU() && g2.IsCPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.IsCPU() && g2.IsCPU())	memcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T));
		}

		template<typename T1, typename T2>
		void Length(DeviceArray<T1>& lhs, DeviceArray<T2>& rhs);


		template void Length(DeviceArray<float>&, DeviceArray<float3>&);
	}
}
