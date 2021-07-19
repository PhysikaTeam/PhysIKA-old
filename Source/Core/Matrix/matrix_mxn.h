#pragma once

#include <memory>

#include "Core/Platform.h"
#include "Core/Matrix/matrix_base.h"
#include "Core/Array/MemoryManager.h"
#include "Core/Vector/vector_nd.h"

namespace PhysIKA {
template <typename T, DeviceType deviceType = DeviceType::CPU>
class MatrixMN : public MatrixBase
{
public:
    MatrixMN(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_nx(0)
        , m_ny(0)
        , m_totalNum(0)
        , m_data(NULL)
        , m_alloc(alloc)
    {
    }

    MatrixMN(int nx, int ny, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
        : m_nx(nx)
        , m_ny(ny)
        , m_totalNum(nx * ny)
        , m_data(NULL)
        , m_alloc(alloc)
    {
        allocMemory();
    }

    MatrixMN(const MatrixMN<T, deviceType>& m);
    MatrixMN(MatrixMN<T, deviceType>&& m);

    /*!
        *    \brief    Should not release data here, call Release() explicitly.
        */
    ~MatrixMN();

    void resize(int nx, int ny);

    void setZeros();

    void release();

    inline T* GetDataPtr()
    {
        return m_data;
    }
    void SetDataPtr(T* _data)
    {
        m_data = _data;
    }

    COMM_FUNC virtual unsigned int rows() const
    {
        return m_nx;
    }
    COMM_FUNC virtual unsigned int cols() const
    {
        return m_ny;
    }

    COMM_FUNC inline const T operator()(unsigned int i, unsigned int j) const
    {
        return m_data[i * m_ny + j];
    }

    COMM_FUNC inline T& operator()(unsigned int i, unsigned int j)
    {
        return m_data[i * m_ny + j];
    }

    COMM_FUNC inline int Index(const int i, const int j)
    {
        return i * m_ny + j;
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

    COMM_FUNC const MatrixMN<T, deviceType> operator+(const MatrixMN<T, deviceType>&) const;
    COMM_FUNC MatrixMN<T, deviceType>& operator+=(const MatrixMN<T, deviceType>&);
    COMM_FUNC const MatrixMN<T, deviceType> operator-(const MatrixMN<T, deviceType>&) const;
    COMM_FUNC MatrixMN<T, deviceType>& operator-=(const MatrixMN<T, deviceType>&);

    COMM_FUNC MatrixMN<T, deviceType>& operator=(const MatrixMN<T, deviceType>&);
    COMM_FUNC MatrixMN<T, deviceType>& operator=(MatrixMN<T, deviceType>&& m);

    COMM_FUNC bool operator==(const MatrixMN<T, deviceType>&) const;
    COMM_FUNC bool operator!=(const MatrixMN<T, deviceType>&) const;

    COMM_FUNC const MatrixMN<T, deviceType> operator*(T) const;
    COMM_FUNC MatrixMN<T, deviceType>& operator*=(T);

    COMM_FUNC const Vectornd<T, deviceType> operator*(const Vectornd<T, deviceType>&) const;
    COMM_FUNC const MatrixMN<T, deviceType> operator*(const MatrixMN<T, deviceType>&) const;
    //COMM_FUNC MatrixMN<T, deviceType>& operator*= (const MatrixMN<T, deviceType> &);

    COMM_FUNC const MatrixMN<T, deviceType> operator/(T) const;
    COMM_FUNC MatrixMN<T, deviceType>& operator/=(T);

    COMM_FUNC const MatrixMN<T, deviceType> operator-(void) const;

    COMM_FUNC const MatrixMN<T, deviceType> transpose() const;

    // set submatrix as matrix m.
    // the left up index of sub matrix is (i,j)
    COMM_FUNC void setSubMatrix(const MatrixMN<T, deviceType>& m, int lui, int luj);

public:
    void allocMemory();

protected:
    int                                        m_nx;
    int                                        m_ny;
    int                                        m_totalNum;
    T*                                         m_data;
    std::shared_ptr<MemoryManager<deviceType>> m_alloc;
};
template <typename T, DeviceType deviceType>
inline MatrixMN<T, deviceType>::MatrixMN(const MatrixMN<T, deviceType>& m)
{
    this->m_nx       = m.m_nx;
    this->m_ny       = m.m_ny;
    this->m_alloc    = m.m_alloc;
    this->m_totalNum = m.m_totalNum;
    this->m_data     = 0;
    this->allocMemory();

    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] = m.m_data[i];
    }
}
template <typename T, DeviceType deviceType>
inline MatrixMN<T, deviceType>::MatrixMN(MatrixMN<T, deviceType>&& m)
{
    *this = std::move(m);
    //this->m_nx = m.m_nx;
    //this->m_ny = m.m_ny;
    //this->m_alloc = m.m_alloc;
    //this->m_totalNum = m.m_totalNum;
    //this->m_data = m.m_data;

    //m.m_nx = 0;
    //m.m_ny = 0;
    //m.m_totalNum = 0;
    //m.m_data = 0;
}
template <typename T, DeviceType deviceType>
inline MatrixMN<T, deviceType>::~MatrixMN()
{
    release();
}
template <typename T, DeviceType deviceType>
inline void MatrixMN<T, deviceType>::resize(int nx, int ny)
{
    assert(nx >= 1 && ny >= 1);
    int n = nx * ny;
    if (n != this->m_totalNum)
    {
        T*  tmp   = this->m_data;
        int tmp_n = this->m_totalNum;

        this->m_nx       = nx;
        this->m_ny       = ny;
        this->m_totalNum = n;
        this->m_data     = 0;
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
inline void MatrixMN<T, deviceType>::setZeros()
{
    m_alloc->initMemory(( void* )m_data, 0, m_totalNum * sizeof(T));
}
template <typename T, DeviceType deviceType>
inline void MatrixMN<T, deviceType>::release()
{
    if (m_data != NULL)
    {
        m_alloc->releaseMemory(( void** )&m_data);
    }

    m_data     = NULL;
    m_nx       = 0;
    m_ny       = 0;
    m_totalNum = 0;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC void MatrixMN<T, deviceType>::setSubMatrix(const MatrixMN<T, deviceType>& m, int lui, int luj)
{
    int n_rows = m.m_nx;
    int n_cols = m.m_ny;

    n_rows = (lui + n_rows) < this->m_nx ? n_rows : this->m_nx - lui;
    n_cols = (luj + n_cols) < this->m_ny ? n_cols : this->m_ny - luj;

    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; ++j)
        {
            //this->m_data(lui + i, luj + j) = m(i, j);
            this->m_data[(lui + i) * m_ny + j] = m(i, j);
        }
    }
    return;
}
template <typename T, DeviceType deviceType>
inline void MatrixMN<T, deviceType>::allocMemory()
{
    //size_t pitch;

    //m_alloc->allocMemory2D((void**)&m_data, pitch, m_nx, m_ny, sizeof(T));
    m_alloc->allocMemory1D(( void** )&m_data, m_totalNum, sizeof(T));

    setZeros();
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator+(const MatrixMN<T, deviceType>& m) const
{
    MatrixMN<T, deviceType> res(m_nx, m_ny);
    for (int i = 0; i < m_totalNum; ++i)
    {
        res.m_data[i] = this->m_data[i] + m.m_data[i];
    }
    return res;
}

template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator+=(const MatrixMN<T, deviceType>& m)
{
    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] += m.m_data[i];
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator-(const MatrixMN<T, deviceType>& m) const
{
    MatrixMN<T, deviceType> res(m_nx, m_ny);
    for (int i = 0; i < m_totalNum; ++i)
    {
        res.m_data[i] = this->m_data[i] - m.m_data[i];
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator-=(const MatrixMN<T, deviceType>& m)
{
    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] -= m.m_data[i];
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator=(const MatrixMN<T, deviceType>& m)
{
    //assert((this->m_nx == m.m_nx) && (this->m_ny == m.m_ny));
    if (this->m_totalNum != (m.m_nx * m.m_ny))
    {
        this->resize(m.m_nx, m.m_ny);
    }
    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] = m.m_data[i];
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator=(MatrixMN<T, deviceType>&& m)
{
    if (this != &m)
    {
        this->m_nx       = m.m_nx;
        this->m_ny       = m.m_ny;
        this->m_alloc    = m.m_alloc;
        this->m_totalNum = m.m_totalNum;
        this->m_data     = m.m_data;

        m.m_nx       = 0;
        m.m_ny       = 0;
        m.m_totalNum = 0;
        m.m_data     = 0;
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC bool MatrixMN<T, deviceType>::operator==(const MatrixMN<T, deviceType>& m) const
{
    assert((this->m_nx == m.m_nx) && (this->m_ny == m.m_ny));
    bool res = true;
    for (int i = 0; i < m_totalNum; ++i)
    {
        if (this->m_data[i] != m.m_data[i])
        {
            res = false;
            break;
        }
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC bool MatrixMN<T, deviceType>::operator!=(const MatrixMN<T, deviceType>& m) const
{
    return !((*this) == m);
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator*(T v) const
{
    MatrixMN<T, deviceType> res(m_nx, m_ny);
    for (int i = 0; i < m_totalNum; ++i)
    {
        res.m_data[i] = this->m_data[i] * v;
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator*=(T v)
{
    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] *= v;
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const Vectornd<T, deviceType> MatrixMN<T, deviceType>::operator*(const Vectornd<T, deviceType>& v) const
{
    assert(this->m_ny == v.size());
    Vectornd<T, deviceType> res(m_nx);
    for (int i = 0; i < m_nx; ++i)
    {
        res[i] = 0;
        for (int j = 0; j < m_ny; ++j)
        {
            res[i] += this->m_data[i * m_ny + j] * v[j];
        }
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator*(const MatrixMN<T, deviceType>& m) const
{
    assert(this->m_ny == m.m_nx);
    MatrixMN<T, deviceType> res(this->m_nx, m.m_ny);
    for (int i = 0; i < this->m_nx; ++i)
    {
        for (int j = 0; j < m.m_ny; ++j)
        {
            res.m_data[i * res.m_ny + j] = 0;
            for (int k = 0; k < m.m_nx; ++k)
            {
                res.m_data[i * res.m_ny + j] += this->m_data[i * this->m_ny + k] * m.m_data[k * m.m_ny + j];
            }
        }
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator/(T v) const
{
    assert(v != 0);
    MatrixMN<T, deviceType> res(m_nx, m_ny);
    for (int i = 0; i < m_totalNum; ++i)
    {
        res[i] = this->m_data[i] / v;
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC MatrixMN<T, deviceType>& MatrixMN<T, deviceType>::operator/=(T v)
{
    assert(v != 0);
    for (int i = 0; i < m_totalNum; ++i)
    {
        this->m_data[i] /= v;
    }
    return *this;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::operator-(void) const
{
    MatrixMN<T, deviceType> res(m_nx, m_ny);
    for (int i = 0; i < m_totalNum; ++i)
    {
        res.m_data[i] = -this->m_data[i];
    }
    return res;
}
template <typename T, DeviceType deviceType>
inline COMM_FUNC const MatrixMN<T, deviceType> MatrixMN<T, deviceType>::transpose() const
{
    MatrixMN<T, deviceType> res(m_ny, m_nx);
    for (int i = 0; i < m_nx; ++i)
    {
        for (int j = 0; j < m_ny; ++j)
        {
            res.m_data[j * res.m_ny + i] = this->m_data[i * this->m_ny + j];
        }
    }
    return res;
}
}  // namespace PhysIKA