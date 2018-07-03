#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/Array2D.h"
#include "Physika_Core/Cuda_Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/

namespace Physika
{
	namespace Function2Pt
	{
		// z = x + y;
		template <typename T>
		void Plus(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x - y;
		template <typename T>
		void Subtract(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x * y;
		template <typename T>
		void Multiply(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x / y;
		template <typename T>
		void Divide(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = a * x + y;
		template <typename T>
		void Saxpy(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr, T alpha);

		template void Plus(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void Plus(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void Plus(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void Subtract(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void Subtract(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void Subtract(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void Multiply(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void Multiply(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void Multiply(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void Divide(DeviceArray<int>&, DeviceArray<int>&, DeviceArray<int>&);
		template void Divide(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&);
		template void Divide(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&);

		template void Saxpy(DeviceArray<float>&, DeviceArray<float>&, DeviceArray<float>&, float);
		template void Saxpy(DeviceArray<double>&, DeviceArray<double>&, DeviceArray<double>&, double);

	};
}
