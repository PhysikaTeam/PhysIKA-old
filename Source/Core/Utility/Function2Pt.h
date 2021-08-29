#pragma once
#include "Core/Array/Array.h"
#include "Core/Array/Array2D.h"
#include "Core/Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/

namespace PhysIKA {
namespace Function2Pt {
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
};  // namespace Function2Pt
}  // namespace PhysIKA
