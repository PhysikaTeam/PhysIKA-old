
#pragma once

//#include "Core/Array/Array.h"
#include "vector_base.h"
#include "math.h"

#include <iostream>
//#include "Core/Array/Array.h"
#include "Core/Platform.h"
#include "Core/Array/MemoryManager.h"
#include <glm/gtx/norm.hpp>

namespace PhysIKA {

template <typename T, DeviceType deviceType = DeviceType::CPU>
class Vectornd : public VectorBase<T>
{

protected:
    void allocMemory()
    {
        if (m_n <= 0)
        {
            m_data = nullptr;
            return;
        }
        m_alloc->allocMemory1D(( void** )&m_data, m_n, sizeof(T));
        //reset();
        this->setZeros();
    }

public:
    Vectornd(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_n(0), m_data(NULL), m_alloc(alloc)
    {
        //this->setZeros();
        this->allocMemory();
    }

    Vectornd(const Vectornd<T, deviceType>& v)
        : m_n(0), m_data(NULL)
    {
        this->m_alloc = v.m_alloc;
        this->m_n     = v.m_n;
        this->allocMemory();
        for (int i = 0; i < this->m_n; ++i)
        {
            this->m_data[i] = v.m_data[i];
        }
    }
    Vectornd(Vectornd<T, deviceType>&& v);

    Vectornd(int num, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_data(NULL)
        , m_n(num)
        , m_alloc(alloc)
    {
        allocMemory();
    }

    ~Vectornd();

    void setZeros();

    void                  resize(int n);
    COMM_FUNC virtual int size() const
    {
        return m_n;
    }
    //virtual int size()const = 0;

    COMM_FUNC inline T& operator()(unsigned int i)
    {
        return m_data[i];
    }
    COMM_FUNC inline const T& operator()(unsigned int i) const
    {
        return m_data[i];
    }
    COMM_FUNC inline T& operator[](unsigned int i)
    {
        return m_data[i];
    }
    COMM_FUNC inline const T& operator[](unsigned int i) const
    {
        return m_data[i];
    }

    Vectornd<T, deviceType> getNegative() const;
    void                    setNegative();

    Vectornd<T, deviceType>& operator=(const Vectornd<T, deviceType>& v);
    Vectornd<T, deviceType>& operator=(Vectornd<T, deviceType>&& v);

    Vectornd<T, deviceType> operator+(const Vectornd<T, deviceType>& v) const;
    Vectornd<T, deviceType> operator-(const Vectornd<T, deviceType>& m) const;
    Vectornd<T, deviceType> operator-(void) const;

    T                       operator*(const Vectornd<T, deviceType>& v) const;
    Vectornd<T, deviceType> operator*(T v) const;

    //Vectornd<T, deviceType> cross(const Vectornd<T, deviceType>& c)const;

    T                             norm() const;
    void                          normalize();
    const Vectornd<T, deviceType> normalized() const;

    bool isZero() const;

    void setSubVector(const Vectornd<T, deviceType>& v, int i);

    //virtual void out()const
    //{
    //    for (int i = 0; i < m_n; ++i)
    //    {
    //        std::cout << m_data[i] << "\t";
    //    }
    //    std::cout << std::endl;
    //}

    DeviceType getDeviceType()
    {
        return deviceType;
    }

    virtual void swap(unsigned int i, unsigned int j);

    void swap(Vectornd<T, deviceType>& arr);

    void release();

private:
    T*                                         m_data = 0;
    int                                        m_n    = 0;
    std::shared_ptr<MemoryManager<deviceType>> m_alloc;
};

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType>::Vectornd(Vectornd<T, deviceType>&& v)
{
    *this = std::move(v);
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType>::~Vectornd()
{
    if (m_data != NULL && m_n != 0)
    {
        m_alloc->releaseMemory(( void** )&m_data);
    }

    m_data = NULL;
    m_n    = 0;
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::resize(int n)
{
    assert(n >= 1);
    if (n != this->m_n)
    {
        T*  tmp   = this->m_data;
        int tmp_n = this->m_n;

        this->m_n    = n;
        this->m_data = 0;
        this->allocMemory();
        tmp_n = tmp_n < n ? tmp_n : n;
        if (NULL != tmp)
        {
            for (int i = 0; i < tmp_n; ++i)
            {
                this->m_data[i] = tmp[i];
            }
            m_alloc->releaseMemory(( void** )&tmp);
        }
    }
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType> Vectornd<T, deviceType>::getNegative() const
{
    Vectornd<T, deviceType> res(m_n);
    for (int i = 0; i < m_n; ++i)
    {
        res.m_data[i] = -this->m_data[i];
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::setNegative()
{
    for (int i = 0; i < m_n; ++i)
    {
        this->m_data[i] = -this->m_data[i];
    }
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType>& Vectornd<T, deviceType>::operator=(const Vectornd<T, deviceType>& v)
{
    // TODO: insert return statement here
    this->m_alloc = v.m_alloc;
    //this->m_n = v.m_n;
    if (this->m_n != v.m_n)
    {
        this->resize(v.m_n);
    }
    for (int i = 0; i < this->m_n; ++i)
    {
        this->m_data[i] = v.m_data[i];
    }
    return *this;
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType>& Vectornd<T, deviceType>::operator=(Vectornd<T, deviceType>&& v)
{
    if (this != &v)
    {
        this->m_alloc = v.m_alloc;
        this->m_n     = v.m_n;
        this->m_data  = v.m_data;

        v.m_n    = 0;
        v.m_data = 0;
    }
    return *this;
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType> Vectornd<T, deviceType>::operator+(const Vectornd<T, deviceType>& v) const
{
    Vectornd<T, deviceType> res(m_n);
    for (int i = 0; i < m_n; ++i)
    {
        res.m_data[i] = this->m_data[i] + v.m_data[i];
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType> Vectornd<T, deviceType>::operator-(const Vectornd<T, deviceType>& v) const
{
    Vectornd<T, deviceType> res(m_n);
    for (int i = 0; i < m_n; ++i)
    {
        res.m_data[i] = this->m_data[i] - v.m_data[i];
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType> Vectornd<T, deviceType>::operator-(void) const
{
    Vectornd<T, deviceType> res(m_n);
    for (int i = 0; i < m_n; ++i)
    {
        res.m_data[i] = -(this->m_data[i]);
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline T Vectornd<T, deviceType>::operator*(const Vectornd<T, deviceType>& v) const
{
    T res = 0;
    for (int i = 0; i < m_n; ++i)
    {
        res = res + this->m_data[i] * v.m_data[i];
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline Vectornd<T, deviceType> Vectornd<T, deviceType>::operator*(T v) const
{
    Vectornd<T, deviceType> res(m_n);
    for (int i = 0; i < m_n; ++i)
    {
        res.m_data[i] = this->m_data[i] * v;
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline T Vectornd<T, deviceType>::norm() const
{
    T sum = 0;
    for (int i = 0; i < m_n; ++i)
    {
        sum += this->m_data[i] * this->m_data[i];
    }

    if (DeviceType::CPU == deviceType)
    {
        return std::sqrt(sum);
    }
    else
    {
        //  ??????? how to calculate sqrt in GPU
        return 0;
    }
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::normalize()
{
    T norm_ = this->norm();
    if (norm_ != 0)
    {
        for (int i = 0; i < m_n; ++i)
        {
            this->m_data /= norm_;
        }
    }
}

template <typename T, DeviceType deviceType>
inline const Vectornd<T, deviceType> Vectornd<T, deviceType>::normalized() const
{
    Vectornd<T, deviceType> res(m_n);
    T                       norm_ = this->norm();
    if (norm_ != 0)
    {
        for (int i = 0; i < m_n; ++i)
        {
            res.m_data[i] = this->m_data[i] / norm_;
        }
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline bool Vectornd<T, deviceType>::isZero() const
{
    T sum = 0;
    for (int i = 0; i < m_n; ++i)
    {
        sum = sum + this->m_data[i] * this->m_data[i];
    }
    return sum < 1e-8;
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::setSubVector(const Vectornd<T, deviceType>& v, int i)
{
    int n = v.m_n;
    n     = (i + n) < this->m_n ? n : (this->m_n - i);

    for (int idx = 0; idx < n; ++idx)
    {
        this->m_data[idx + i] = v(idx);
    }
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::setZeros()
{
    if (m_n <= 0)
        return;
    m_alloc->initMemory(( void* )m_data, 0, m_n * sizeof(T));
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::swap(unsigned int i, unsigned int j)
{
    T v             = this->m_data[i];
    this->m_data[i] = this->m_data[j];
    this->m_data[j] = v;
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::swap(Vectornd<T, deviceType>& arr)
{
    assert(m_n == arr.m_n);
    T* tp      = arr.m_data;
    arr.m_data = m_data;
    m_data     = tp;
}

template <typename T, DeviceType deviceType>
inline void Vectornd<T, deviceType>::release()
{
    if (m_data != NULL)
    {
        m_alloc->releaseMemory(( void** )&m_data);
    }

    m_data = NULL;
    m_n    = 0;
}

}  // namespace PhysIKA
