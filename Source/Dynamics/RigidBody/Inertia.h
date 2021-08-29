#pragma once

//#include "Core/Matrix/matrix_base.h"
#include "Core/Vector/vector_3d.h"
#include "SpatialVector.h"

namespace PhysIKA {
template <typename T>
class Inertia  // : public MatrixBase<T>
{
public:
    Inertia()
        : m_inertiaDiagonal(0, 0, 0), m_mass(0) {}
    Inertia(const T& mass, const Vector<T, 3>& inertaiDiagonal)
        : m_inertiaDiagonal(inertaiDiagonal), m_mass(mass) {}

    virtual unsigned int rows() const
    {
        return 6;
    }
    virtual unsigned int cols() const
    {
        return 6;
    }

    //virtual T& operator()(unsigned int i, unsigned int j);
    virtual const T operator()(unsigned int i, unsigned int j) const;

    const Vector<T, 3>& getInertiaDiagonal() const
    {
        return m_inertiaDiagonal;
    }
    Vector<T, 3>& getInertiaDiagonal()
    {
        return m_inertiaDiagonal;
    }

    const T getMass() const
    {
        return m_mass;
    }

    void setInertiaDiagonal(const Vector<T, 3>& inertiaDiagonal)
    {
        m_inertiaDiagonal = inertiaDiagonal;
    }
    void setMass(const T& mass)
    {
        m_mass = mass;
    }

    const SpatialVector<float> operator*(const SpatialVector<float>& v) const;
    const Inertia<float>       operator+(const Inertia<T>& inertia) const;

    void getTensor(MatrixMN<T>& m) const;

    Inertia<T> getInverse() const
    {
        Inertia<T> inv;
        inv.m_mass               = m_mass == 0 ? 0 : 1.0 / m_mass;
        inv.m_inertiaDiagonal[0] = m_inertiaDiagonal[0] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[0];
        inv.m_inertiaDiagonal[1] = m_inertiaDiagonal[1] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[1];
        inv.m_inertiaDiagonal[2] = m_inertiaDiagonal[2] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[2];
        return inv;
    }
    void getInverse(Inertia<T>& inv) const
    {
        inv.m_mass               = m_mass == 0 ? 0 : 1.0 / m_mass;
        inv.m_inertiaDiagonal[0] = m_inertiaDiagonal[0] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[0];
        inv.m_inertiaDiagonal[1] = m_inertiaDiagonal[1] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[1];
        inv.m_inertiaDiagonal[2] = m_inertiaDiagonal[2] == 0.0f ? 0 : 1.0 / m_inertiaDiagonal[2];
    }

public:
    Vector<T, 3> m_inertiaDiagonal;
    T            m_mass;
};

template <typename T>
inline const T Inertia<T>::operator()(unsigned int i, unsigned int j) const
{
    return (i != j) ? 0 : ((i < 3) ? (m_inertiaDiagonal[i]) : m_mass);
}

template <typename T>
inline const SpatialVector<float> Inertia<T>::operator*(const SpatialVector<float>& v2) const
{
    SpatialVector<float> res;

    res[0] = v2[0] * m_inertiaDiagonal[0];
    res[1] = v2[1] * m_inertiaDiagonal[1];
    res[2] = v2[2] * m_inertiaDiagonal[2];
    res[3] = v2[3] * m_mass;
    res[4] = v2[4] * m_mass;
    res[5] = v2[5] * m_mass;

    return res;
}
template <typename T>
inline void Inertia<T>::getTensor(MatrixMN<T>& m) const
{
    m.resize(6, 6);
    m.setZeros();

    m(0, 0) = m_inertiaDiagonal[0];
    m(1, 1) = m_inertiaDiagonal[1];
    m(2, 2) = m_inertiaDiagonal[2];
    m(3, 3) = m_mass;
    m(4, 4) = m_mass;
    m(5, 5) = m_mass;
}
}  // namespace PhysIKA