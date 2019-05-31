#pragma once
#include "Physika_Core/Array/Array.h"
#include "Physika_Core/Array/Array2D.h"
#include "Physika_Core/Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/

namespace Physika
{
	namespace Function2Pt
	{
		// z = x + y;
		template <typename T>
		void plus(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x - y;
		template <typename T>
		void subtract(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x * y;
		template <typename T>
		void multiply(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x / y;
		template <typename T>
		void divide(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = a * x + y;
		template <typename T>
		void saxpy(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr, T alpha);

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

	};
}
