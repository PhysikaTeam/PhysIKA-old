#pragma once
#include "Core/Array/Array.h"
#include "Core/Array/DynamicArray.h"
#include "Core/Array/Array2D.h"
#include "Core/Array/Array3D.h"
#include "Core/Array/ArrayPitch2D.h"
#include "Core/Utility/cuda_utilities.h"
/*
*  This file implements all one-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/
namespace PhysIKA {
namespace Function1Pt {
template <typename T, DeviceType dType1, DeviceType dType2>
void copy(Array<T, dType1>& arr1, Array<T, dType2>& arr2)
{
    assert(arr1.size() == arr2.size());
    int totalNum = arr1.size();
    if (arr1.isGPU() && arr2.isGPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else if (arr1.isCPU() && arr2.isGPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
    }
    else if (arr1.isGPU() && arr2.isCPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
    }
    else if (arr1.isCPU() && arr2.isCPU())
        memcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T));
}

template <typename T, DeviceType dType1, DeviceType dType2>
void copy(DynArray<T, dType1>& arr1, DynArray<T, dType2>& arr2)
{
    assert(arr1.size() == arr2.size());
    int totalNum = arr1.size();
    if (arr1.isGPU() && arr2.isGPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else if (arr1.isCPU() && arr2.isGPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
    }
    else if (arr1.isGPU() && arr2.isCPU())
    {
        cuSafeCall(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
    }
    else if (arr1.isCPU() && arr2.isCPU())
        memcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T));
}

template <typename T, DeviceType deviceType>
void copy(Array<T, deviceType>& arr, std::vector<T>& vec)
{
    assert(vec.size() == arr.size());
    int totalNum = arr.size();
    switch (deviceType)
    {
        case CPU:
            memcpy(arr.begin(), &vec[0], totalNum * sizeof(T));
            break;
        case GPU:
            cuSafeCall(cudaMemcpy(arr.begin(), &vec[0], totalNum * sizeof(T), cudaMemcpyHostToDevice));
            break;
        default:
            break;
    }
}

template <typename T, DeviceType deviceType>
void copy(DynArray<T, deviceType>& arr, std::vector<T>& vec)
{
    assert(vec.size() == arr.size());
    int totalNum = arr.size();
    switch (deviceType)
    {
        case CPU:
            memcpy(arr.begin(), &vec[0], totalNum * sizeof(T));
            break;
        case GPU:
            cuSafeCall(cudaMemcpy(arr.begin(), &vec[0], totalNum * sizeof(T), cudaMemcpyHostToDevice));
            break;
        default:
            break;
    }
}

template <typename T, DeviceType deviceType>
void copy(std::vector<T>& vec, Array<T, deviceType>& arr)
{
    assert(vec.size() == arr.size());
    int totalNum = arr.size();
    switch (deviceType)
    {
        case CPU:
            memcpy(&vec[0], arr.begin(), totalNum * sizeof(T));
            break;
        case GPU:
            cuSafeCall(cudaMemcpy(&vec[0], arr.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
            break;
        default:
            break;
    }
}

template <typename T, DeviceType deviceType>
void copy(std::vector<T>& vec, DynArray<T, deviceType>& arr)
{
    assert(vec.size() == arr.size());
    int totalNum = arr.size();
    switch (deviceType)
    {
        case CPU:
            memcpy(&vec[0], arr.begin(), totalNum * sizeof(T));
            break;
        case GPU:
            cuSafeCall(cudaMemcpy(&vec[0], arr.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
            break;
        default:
            break;
    }
}

template <typename T, DeviceType dType1, DeviceType dType2>
void copy(Array2D<T, dType1>& g1, Array2D<T, dType2>& g2)
{
    assert(g1.Size() == g2.Size() && g1.Nx() == g2.Nx() && g1.Ny() == g2.Ny());
    int width  = g1.Nx() * sizeof(T);
    int height = g1.Ny();
    if (g1.IsGPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyDeviceToDevice));
    }
    else if (g1.IsCPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyDeviceToHost));
    }
    else if (g1.IsGPU() && g2.IsCPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyHostToDevice));
    }
    else if (g1.IsCPU() && g2.IsCPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyHostToHost));
    }
}

template <typename T, DeviceType dType1, DeviceType dType2>
void copy(ArrayPitch2D<T, dType1>& g1, ArrayPitch2D<T, dType2>& g2)
{
    assert(g1.Size() == g2.Size() && g1.Nx() == g2.Nx() && g1.Ny() == g2.Ny());
    int width  = g1.Nx() * sizeof(T);
    int height = g1.Ny();
    if (g1.IsGPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyDeviceToDevice));
    }
    else if (g1.IsCPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyDeviceToHost));
    }
    else if (g1.IsGPU() && g2.IsCPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyHostToDevice));
    }
    else if (g1.IsCPU() && g2.IsCPU())
    {
        cuSafeCall(cudaMemcpy2D(g1.GetDataPtr(), g1.Pitch() * sizeof(T), g2.GetDataPtr(), g2.Pitch() * sizeof(T), width, height, cudaMemcpyHostToHost));
    }
}

template <typename T, DeviceType dType1, DeviceType dType2>
void copy(Array3D<T, dType1>& g1, Array3D<T, dType1>& g2)
{
    assert(g1.Size() == g2.Size() && g1.Nx() == g2.Nx() && g1.Ny() == g2.Ny() && g1.Nz() == g2.Nz());
    int totalNum = g1.Size();
    if (g1.IsGPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else if (g1.IsCPU() && g2.IsGPU())
    {
        cuSafeCall(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
    }
    else if (g1.IsGPU() && g2.IsCPU())
    {
        cuSafeCall(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
    }
    else if (g1.IsCPU() && g2.IsCPU())
        memcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T));
}

template <typename T1, typename T2>
void Length(DeviceArray<T1>& lhs, DeviceArray<T2>& rhs);

}  // namespace Function1Pt
}  // namespace PhysIKA
