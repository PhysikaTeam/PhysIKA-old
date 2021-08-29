#pragma once
#include <assert.h>
//#include <cuda_runtime.h>
#include "Core/Platform.h"
#include "MemoryManager.h"
namespace PhysIKA {

#define INVALID -1

template <typename T, DeviceType deviceType = DeviceType::GPU>
class ArrayPitch2D
{
public:
    ArrayPitch2D(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_nx(0)
        , m_ny(0)
        , m_pitch(0)
        , m_totalNum(0)
        , m_data(NULL)
        , m_alloc(alloc){};

    ArrayPitch2D(int nx, int ny, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_nx(nx)
        , m_ny(ny)
        , m_pitch(nx)
        , m_totalNum(nx * ny)
        , m_data(NULL)
        , m_alloc(alloc)
    {
        AllocMemory();
    };

    /*!
        *    \brief    Should not release data here, call Release() explicitly.
        */
    ~ArrayPitch2D(){};

    void Resize(int nx, int ny);

    void Reset();

    void Release();

    inline T* GetDataPtr()
    {
        return m_data;
    }
    void SetDataPtr(T* _data)
    {
        m_data = _data;
    }

    COMM_FUNC inline int Nx()
    {
        return m_nx;
    }
    COMM_FUNC inline int Ny()
    {
        return m_ny;
    }
    COMM_FUNC inline size_t Pitch()
    {
        return m_pitch;
    }

    COMM_FUNC inline T operator()(const int i, const int j) const
    {
        return m_data[i + j * m_pitch];
    }

    COMM_FUNC inline T& operator()(const int i, const int j)
    {
        return m_data[i + j * m_pitch];
    }

    COMM_FUNC inline int Index(const int i, const int j)
    {
        return i + j * m_pitch;
    }

    COMM_FUNC inline T operator[](const int id) const
    {
        return m_data[id];
    }

    COMM_FUNC inline T& operator[](const int id)
    {
        return m_data[id];
    }

    COMM_FUNC inline int Size()
    {
        return m_totalNum;
    }
    COMM_FUNC inline bool IsCPU()
    {
        return deviceType;
    }
    COMM_FUNC inline bool IsGPU()
    {
        return deviceType;
    }

public:
    void AllocMemory();

private:
    int                                        m_nx;
    int                                        m_ny;
    int                                        m_totalNum;
    size_t                                     m_pitch;
    T*                                         m_data;
    std::shared_ptr<MemoryManager<deviceType>> m_alloc;
};

template <typename T, DeviceType deviceType>
void ArrayPitch2D<T, deviceType>::Resize(int nx, int ny)
{
    if (NULL != m_data)
        Release();
    m_nx = nx;
    m_ny = ny;
    //m_totalNum = m_nx*m_ny;
    AllocMemory();
}

template <typename T, DeviceType deviceType>
void ArrayPitch2D<T, deviceType>::Reset()
{
    //         switch (deviceType)
    //         {
    //         case CPU:
    //             memset((void*)m_data, 0, m_totalNum * sizeof(T));
    //             break;
    //         case GPU:
    //             cudaMemset(m_data, 0, m_totalNum * sizeof(T));
    //             break;
    //         default:
    //             break;
    //         }

    m_alloc->initMemory(( void* )m_data, 0, m_totalNum * sizeof(T));
}

template <typename T, DeviceType deviceType>
void ArrayPitch2D<T, deviceType>::Release()
{
    if (m_data != NULL)
    {
        //             switch (deviceType)
        //             {
        //             case CPU:
        //                 delete[]m_data;
        //                 break;
        //             case GPU:
        //                 (cudaFree(m_data));
        //                 break;
        //             default:
        //                 break;
        //             }

        m_alloc->releaseMemory(( void** )&m_data);
    }

    m_data     = NULL;
    m_nx       = 0;
    m_ny       = 0;
    m_pitch    = 0;
    m_totalNum = 0;
}

template <typename T, DeviceType deviceType>
void ArrayPitch2D<T, deviceType>::AllocMemory()
{
    //         switch (deviceType)
    //         {
    //         case CPU:
    //             m_data = new T[m_totalNum];
    //             break;
    //         case GPU:
    //             (cudaMalloc((void**)&m_data, m_totalNum * sizeof(T)));
    //             break;
    //         default:
    //             break;
    //         }
    //size_t pitch;

    // Note: y is row direction, and x is column direction
    // To get element(i,j), use: m_data[j*m_pitch + i]
    m_alloc->allocMemory2D(( void** )&m_data, m_pitch, m_ny, m_nx, sizeof(T));
    m_pitch /= sizeof(T);
    m_totalNum = m_ny * m_pitch;

    Reset();
}

template <typename T>
using HostArrayPitch2D = ArrayPitch2D<T, DeviceType::CPU>;

template <typename T>
using DeviceArrayPitch2D = ArrayPitch2D<T, DeviceType::GPU>;

using DeviceArrayPitch2D1f = DeviceArrayPitch2D<float>;
using DeviceArrayPitch2D2f = DeviceArrayPitch2D<float2>;
using DeviceArrayPitch2D3f = DeviceArrayPitch2D<float3>;
using DeviceArrayPitch2D4f = DeviceArrayPitch2D<float4>;

template <typename T, DeviceType deviceType1, DeviceType deviceType2>
void CopyFrom2D_external(ArrayPitch2D<T, deviceType1>& arr1, ArrayPitch2D<T, deviceType2>& arr2)
{
    assert(/*arr1.Size() == arr2.Size() &&*/ arr1.Nx()() == arr2.Nx() && arr2.Ny() == arr2.Ny());

    if (deviceType1 == DeviceType::CPU && deviceType2 == DeviceType::CPU)
    {
        memcpy(( void* )(arr1.GetDataPtr()), ( void* )(arr2.GetDataPtr()), sizeof(T) * arr1.Size());
    }
    else if (deviceType1 == DeviceType::CPU && deviceType2 == DeviceType::GPU)
    {
        int sizeT = sizeof(T);
        cudaMemcpy2D(arr1.GetDataPtr(), arr1.Pitch() * sizeT, arr2.GetDataPtr(), arr2.Pitch() * sizeT, arr1.Nx() * sizeT, arr1.Ny(), cudaMemcpyDeviceToHost);
    }
    else if (deviceType1 == DeviceType::GPU && deviceType2 == DeviceType::CPU)
    {
        int sizeT = sizeof(T);
        cudaMemcpy2D(arr1.GetDataPtr(), arr1.Pitch() * sizeT, arr2.GetDataPtr(), arr2.Pitch() * sizeT, arr1.Nx() * sizeT, arr1.Ny(), cudaMemcpyHostToDevice);
    }
    else if (deviceType1 == DeviceType::GPU && deviceType2 == DeviceType::GPU)
    {
        int sizeT = sizeof(T);
        cudaMemcpy2D(arr1.GetDataPtr(), arr1.Pitch() * sizeT, arr2.GetDataPtr(), arr2.Pitch() * sizeT, arr1.Nx() * sizeT, arr1.Ny(), cudaMemcpyDeviceToDevice);
    }
}

}  // namespace PhysIKA
