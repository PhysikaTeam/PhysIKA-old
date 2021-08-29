#pragma once

#include <iostream>
#include <glm/vec3.hpp>
#include "Core/Platform.h"
#include "Dynamics/RigidBody/SpatialVector.h"

#include "Core/Matrix/matrix_base.h"
#include <glm/gtx/norm.hpp>

namespace PhysIKA {
template <typename T>
class JointSpaceBase : public MatrixBase
{
public:
    COMM_FUNC virtual unsigned int rows() const
    {
        return 6;
    }
    //virtual unsigned int cols() const { return Dof; }

    COMM_FUNC virtual const SpatialVector<T> operator*(const VectorBase<T>& q) const
    {
        return SpatialVector<T>();
    }
    COMM_FUNC virtual const SpatialVector<T> mul(const T* q) const
    {
        return SpatialVector<T>();
    }

    COMM_FUNC virtual void transposeMul(const SpatialVector<T>& v, T* res) const {}

    COMM_FUNC virtual int dof() const
    {
        return 0;
    }
    COMM_FUNC virtual const SpatialVector<T>* getBases() const = 0;
    COMM_FUNC virtual SpatialVector<T>*       getBases()       = 0;

    COMM_FUNC virtual T&      operator()(unsigned int, unsigned int)       = 0;
    COMM_FUNC virtual const T operator()(unsigned int, unsigned int) const = 0;
};

template <typename T, unsigned int Dof>
class JointSpace : public JointSpaceBase<T>  //public MatrixBase<T>
{
public:
    JointSpace() {}

    //virtual unsigned int rows() const { return 6; }
    COMM_FUNC virtual unsigned int cols() const
    {
        return Dof;
    }

    COMM_FUNC virtual T&      operator()(unsigned int i, unsigned int j);
    COMM_FUNC virtual const T operator()(unsigned int i, unsigned int j) const;

    COMM_FUNC virtual const SpatialVector<T> operator*(const VectorBase<T>& q) const;
    COMM_FUNC virtual const SpatialVector<T> mul(const T* q) const;

    /// transpose multiply. Transform a vector from 6d space to  joint space.
    COMM_FUNC virtual void transposeMul(const SpatialVector<T>& v, T* res) const;

    COMM_FUNC virtual int dof() const
    {
        return Dof;
    }
    COMM_FUNC virtual const SpatialVector<T>* getBases() const
    {
        return m_data;
    }
    COMM_FUNC virtual SpatialVector<T>* getBases()
    {
        return m_data;
    }

    //const
private:
    SpatialVector<T> m_data[Dof];
};

template <typename T, unsigned int Dof>
COMM_FUNC inline T& JointSpace<T, Dof>::operator()(unsigned int i, unsigned int j)
{
    return m_data[j][i];
}
template <typename T, unsigned int Dof>
COMM_FUNC inline const T JointSpace<T, Dof>::operator()(unsigned int i, unsigned int j) const
{
    // TODO: insert return statement here
    return m_data[j][i];
}

template <typename T, unsigned int Dof>
COMM_FUNC inline const SpatialVector<T> JointSpace<T, Dof>::operator*(const VectorBase<T>& q) const
{
    SpatialVector<T> vj(0, 0, 0, 0, 0, 0);
    int              n = q.size();
    for (int i = 0; i < n; ++i)
    {
        vj += (m_data[i] * q[i]);
    }
    return vj;
}
template <typename T, unsigned int Dof>
COMM_FUNC inline const SpatialVector<T> JointSpace<T, Dof>::mul(const T* q) const
{
    SpatialVector<T> vj(0, 0, 0, 0, 0, 0);
    for (int i = 0; i < Dof; ++i)
    {
        vj += (m_data[i] * (q[i]));
    }
    return vj;
}
template <typename T, unsigned int Dof>
COMM_FUNC inline void JointSpace<T, Dof>::transposeMul(const SpatialVector<T>& v, T* res) const
{
    for (int i = 0; i < Dof; ++i)
    {
        res[i] = (this->m_data[i]) * v;
    }
}
}  // namespace PhysIKA