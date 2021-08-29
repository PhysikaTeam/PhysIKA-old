#pragma once
#include <cassert>
#include <vector>
//#include <cuda_runtime.h>
#include <memory>
#include "Core/Platform.h"
#include "Core/Array/MemoryManager.h"
//#include "Core/Utility/Function1Pt.h"

namespace PhysIKA {

/*!
    *    \class    Array
    *    \brief    This class is designed to be elegant, so it can be directly passed to GPU as parameters.
    */
template <typename T, DeviceType deviceType = DeviceType::GPU>
class DynArray
{
public:
    DynArray(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_data(NULL)
        , m_totalNum(0)
        , m_capability(0)
        , m_alloc(alloc){};

    DynArray(int num, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_data(NULL)
        , m_totalNum(num)
        , m_capability(( int )(num * 1.2))
        , m_alloc(alloc)
    {
        allocMemory();
    }

    /*!
        *    \brief    Should not release data here, call Release() explicitly.
        */
    ~DynArray(){};

    void resize(int n);

    void reserve(int n);

    /*!
        *    \brief    Clear all data to zero.
        */
    void reset();

    /*!
        *    \brief    Free allocated memory.    Should be called before the object is deleted.
        */
    void release();

    COMM_FUNC inline T* begin()
    {
        return m_data;
    }

    DeviceType getDeviceType()
    {
        return deviceType;
    }

    void Swap(DynArray<T, deviceType>& arr)
    {
        assert(m_totalNum == arr.size());
        //             T* tp = arr.GetDataPtr();
        //             arr.SetDataPtr(m_data);
        T* tp      = arr.m_data;
        arr.m_data = m_data;
        m_data     = tp;
    }

    COMM_FUNC inline T& operator[](unsigned int id)
    {
        return m_data[id];
    }

    COMM_FUNC inline T operator[](unsigned int id) const
    {
        return m_data[id];
    }

    COMM_FUNC inline int size()
    {
        return m_totalNum;
    }
    COMM_FUNC inline int capability()
    {
        return m_capability;
    }

    COMM_FUNC inline bool isCPU()
    {
        return deviceType == DeviceType::CPU;
    }
    COMM_FUNC inline bool isGPU()
    {
        return deviceType == DeviceType::GPU;
    }
    COMM_FUNC inline bool isEmpty()
    {
        return m_data == NULL;
    }

protected:
    void allocMemory();

private:
    T*  m_data;
    int m_totalNum;
    int m_capability = 0;

    std::shared_ptr<MemoryManager<deviceType>> m_alloc;
};

template <typename T, DeviceType deviceType>
void DynArray<T, deviceType>::resize(const int n)
{
    //        assert(n >= 1);
    if (n <= 0)
    {
        m_totalNum = 0;
        return;
    }
    if (m_capability < n)
    {
        this->reserve(( int )(n * 1.2));
    }

    if (m_totalNum != n)
    {
        m_totalNum = n;
    }
}

template <typename T, DeviceType deviceType>
inline void DynArray<T, deviceType>::reserve(int n)
{
    if (m_capability < n && n > 0)
    {

        T* tmpdata = m_data;
        m_alloc->allocMemory1D(( void** )&m_data, n, sizeof(T));

        if (tmpdata)
        {
            m_alloc->copyMemory1D(m_data, tmpdata, m_capability, sizeof(T));
            m_alloc->releaseMemory(( void** )&tmpdata);
            tmpdata = 0;
        }
        m_capability = n;
    }
}

template <typename T, DeviceType deviceType>
void DynArray<T, deviceType>::release()
{
    if (m_data != NULL)
    {
        m_alloc->releaseMemory(( void** )&m_data);
    }

    m_data       = NULL;
    m_totalNum   = 0;
    m_capability = 0;
}

template <typename T, DeviceType deviceType>
void DynArray<T, deviceType>::allocMemory()
{
    m_alloc->allocMemory1D(( void** )&m_data, m_capability, sizeof(T));

    reset();
}

template <typename T, DeviceType deviceType>
void DynArray<T, deviceType>::reset()
{
    m_alloc->initMemory(( void* )m_data, 0, m_capability * sizeof(T));
}

template <typename T>
using HostDArray = DynArray<T, DeviceType::CPU>;

template <typename T>
using DeviceDArray = DynArray<T, DeviceType::GPU>;
}  // namespace PhysIKA
