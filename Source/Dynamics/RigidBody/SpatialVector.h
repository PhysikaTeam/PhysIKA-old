#pragma once

#include <iostream>
#include <glm/vec3.hpp>
#include "Core/Platform.h"

#include "Core/Vector/vector_base.h"
#include <glm/gtx/norm.hpp>

namespace PhysIKA {

template <typename T>
class SpatialVector : public VectorBase<T>
{
public:
    COMM_FUNC SpatialVector()
        : m_angular(0, 0, 0), m_linear(0, 0, 0) {}

    COMM_FUNC SpatialVector(const T& ang1, const T& ang2, const T& ang3, const T& lin1, const T& lin2, const T& lin3)
        : m_angular(ang1, ang2, ang3), m_linear(lin1, lin2, lin3) {}

    COMM_FUNC SpatialVector(const glm::tvec3<T>& angular, const glm::tvec3<T>& linear)
        : m_angular(angular), m_linear(linear) {}

    COMM_FUNC SpatialVector(const Vector3f& angular, const Vector3f& linear)
        : m_angular(angular[0], angular[1], angular[2]), m_linear(linear[0], linear[1], linear[2]) {}

    COMM_FUNC SpatialVector(const SpatialVector<T>& v)
        : m_angular(v.m_angular), m_linear(v.m_linear) {}

    COMM_FUNC void set(const T& ang1, const T& ang2, const T& ang3, const T& lin1, const T& lin2, const T& lin3);

    COMM_FUNC int size() const
    {
        return 6;
    }

    COMM_FUNC T&    operator[](unsigned int i);
    COMM_FUNC const T& operator[](unsigned int i) const;

    COMM_FUNC const SpatialVector<T> operator+(const SpatialVector<T>& v) const;
    COMM_FUNC SpatialVector<T>& operator+=(const SpatialVector<T>& v);
    COMM_FUNC const SpatialVector<T> operator-(const SpatialVector<T>& v) const;
    COMM_FUNC SpatialVector<T>& operator-=(const SpatialVector<T>& v);

    COMM_FUNC const SpatialVector<T> operator-(void) const;

    //COMM_FUNC const SpatialVector<T> operator* (const SpatialVector<T> & v) const;
    //COMM_FUNC SpatialVector<T>& operator*= (const SpatialVector<T> & v);
    //COMM_FUNC const SpatialVector<T> operator/ (const SpatialVector<T> & v) const;
    //COMM_FUNC SpatialVector<T>& operator/= (const SpatialVector<T> & v);

    COMM_FUNC SpatialVector<T>& operator=(const SpatialVector<T>& v);

    COMM_FUNC bool operator==(const SpatialVector<T>& v) const;
    COMM_FUNC bool operator!=(const SpatialVector<T>& v) const;

    COMM_FUNC const SpatialVector<T> operator*(T v) const;
    COMM_FUNC const SpatialVector<T> operator-(T v) const;
    COMM_FUNC const SpatialVector<T> operator+(T v) const;
    COMM_FUNC const SpatialVector<T> operator/(T v) const;

    COMM_FUNC SpatialVector<T>& operator+=(T v);
    COMM_FUNC SpatialVector<T>& operator-=(T v);
    COMM_FUNC SpatialVector<T>& operator*=(T v);
    COMM_FUNC SpatialVector<T>& operator/=(T v);

    // Product between MOTION vector and  FORCE vector
    // Usage: power of a force.
    COMM_FUNC T operator*(const SpatialVector<T>& v) const;

    COMM_FUNC T norm() const;
    COMM_FUNC T normSquared() const;
    COMM_FUNC SpatialVector<T>& normalize();
    //COMM_FUNC SpatialVector<T> cross(const SpatialVector<T> &) const;
    //COMM_FUNC T dot(const SpatialVector<T>&) const;

    COMM_FUNC const SpatialVector<T> crossM(const SpatialVector<T>& vec) const;
    COMM_FUNC const SpatialVector<T> crossF(const SpatialVector<T>& vec) const;

public:
    glm::tvec3<T> m_angular;
    glm::tvec3<T> m_linear;
};

template <typename T>
inline COMM_FUNC void SpatialVector<T>::set(const T& ang1, const T& ang2, const T& ang3, const T& lin1, const T& lin2, const T& lin3)
{
    m_angular[0] = ang1;
    m_angular[1] = ang2;
    m_angular[2] = ang3;

    m_linear[0] = lin1;
    m_linear[1] = lin2;
    m_linear[2] = lin3;
}

template <typename T>
inline COMM_FUNC T& SpatialVector<T>::operator[](unsigned int i)
{
    if (i < 3)
    {
        return m_angular[i];
    }
    else
    {
        return m_linear[i - 3];
    }
}

template <typename T>
inline COMM_FUNC const T& SpatialVector<T>::operator[](unsigned int i) const
{
    if (i < 3)
    {
        return m_angular[i];
    }
    else
    {
        return m_linear[i - 3];
    }
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator+(const SpatialVector<T>& v) const
{
    SpatialVector<T> res(*this);
    res.m_angular += v.m_angular;
    res.m_linear += v.m_linear;
    return res;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator+=(const SpatialVector<T>& v)
{
    this->m_angular += v.m_angular;
    this->m_linear += v.m_linear;
    return *this;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator-(const SpatialVector<T>& v) const
{
    SpatialVector<T> res(*this);
    res.m_angular -= v.m_angular;
    res.m_linear -= v.m_linear;

    return res;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator-=(const SpatialVector<T>& v)
{
    this->m_angular -= v.m_angular;
    this->m_linear -= v.m_linear;
    return *this;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator-(void) const
{
    SpatialVector<T> res;
    res.m_angular = -this->m_angular;
    res.m_linear  = -this->m_linear;
    return res;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator=(const SpatialVector<T>& v)
{
    this->m_angular = v.m_angular;
    this->m_linear  = v.m_linear;
    return *this;
}

template <typename T>
inline COMM_FUNC bool SpatialVector<T>::operator==(const SpatialVector<T>& v) const
{
    return (this->m_angular == v.m_angular) && (this->m_linear == v.m_linear);
}

template <typename T>
inline COMM_FUNC bool SpatialVector<T>::operator!=(const SpatialVector<T>& v) const
{
    return (this->m_angular != v.m_angular) || (this->m_linear != v.m_linear);
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator*(T v) const
{
    SpatialVector<T> res(*this);
    res.m_angular *= v;
    res.m_linear *= v;
    return res;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator-(T v) const
{
    SpatialVector<T> res(*this);
    res.m_angular -= v;
    res.m_linear -= v;
    return res;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator+(T v) const
{
    SpatialVector<T> res(*this);
    res.m_angular += v;
    res.m_linear += v;
    return res;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::operator/(T v) const
{
    assert(v != 0);
    SpatialVector<T> res(*this);
    res.m_angular /= v;
    res.m_linear /= v;
    return res;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator+=(T v)
{
    this->m_angular += v;
    this->m_linear += v;
    return *this;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator-=(T v)
{
    this->m_angular -= v;
    this->m_linear -= v;
    return *this;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator*=(T v)
{
    this->m_angular *= v;
    this->m_linear *= v;
    return *this;
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::operator/=(T v)
{
    assert(v != 0);
    this->m_angular /= v;
    this->m_linear /= v;
    return *this;
}

template <typename T>
inline COMM_FUNC T SpatialVector<T>::operator*(const SpatialVector<T>& v) const
{
    return glm::dot(this->m_angular, v.m_angular) + glm::dot(this->m_linear, v.m_linear);
}

template <typename T>
inline COMM_FUNC T SpatialVector<T>::norm() const
{
    return glm::sqrt(glm::length2(this->m_angular) + glm::length2(this->m_linear));
}

template <typename T>
inline COMM_FUNC T SpatialVector<T>::normSquared() const
{
    return glm::length2(this->m_angular) + glm::length2(this->m_linear);
}

template <typename T>
inline COMM_FUNC SpatialVector<T>& SpatialVector<T>::normalize()
{
    T norm_ = this->norm();
    if (norm_ != 0)
    {
        this->m_angular /= norm_;
        this->m_linear /= norm_;
    }
    return *this;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::crossM(const SpatialVector<T>& vec) const
{
    SpatialVector<T> res;
    res.m_angular = glm::cross(this->m_angular, vec.m_angular);
    res.m_linear  = glm::cross(this->m_angular, vec.m_linear) + glm::cross(this->m_linear, vec.m_angular);

    return res;
}

template <typename T>
inline COMM_FUNC const SpatialVector<T> SpatialVector<T>::crossF(const SpatialVector<T>& vec) const
{
    SpatialVector<T> res;
    res.m_angular = glm::cross(this->m_angular, vec.m_angular) + glm::cross(this->m_linear, vec.m_linear);
    res.m_linear  = glm::cross(this->m_angular, vec.m_linear);

    return res;
}

}  // namespace PhysIKA

//#include "SpatialVector.inl"
