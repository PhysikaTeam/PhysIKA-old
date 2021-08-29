#include "Dynamics/RigidBody/BoxAABB3d.h"

namespace PhysIKA {
template <typename T>
PhysIKA::BoxAABB3d<T>::BoxAABB3d()
{
    m_l[0] = 0;
    m_l[1] = 0;
    m_l[2] = 0;

    m_u[0] = 0;
    m_u[1] = 0;
    m_u[2] = 0;
}

template <typename T>
BoxAABB3d<T>::BoxAABB3d(T lx, T ly, T lz, T ux, T uy, T uz)
{
    m_l[0] = lx;
    m_l[1] = ly;
    m_l[2] = lz;

    m_u[0] = ux;
    m_u[1] = uy;
    m_u[2] = uz;
}

template <typename T>
BoxAABB3d<T>::BoxAABB3d(T* pl, T* pu)
{
    m_l[0] = pl[0];
    m_l[1] = pl[1];
    m_l[2] = pl[2];

    m_u[0] = pu[0];
    m_u[1] = pu[1];
    m_u[2] = pu[2];
}

template <typename T>
bool BoxAABB3d<T>::isIntersect(const BoxAABB3d<T>& b) const
{
    if ((this->m_l[0] > b.m_u[0]) || (this->m_u[0] < b.m_l[0]))
    {
        return false;
    }
    if ((this->m_l[1] > b.m_u[1]) || (this->m_u[1] < b.m_l[1]))
    {
        return false;
    }
    if ((this->m_l[2] > b.m_u[2]) || (this->m_u[2] < b.m_l[2]))
    {
        return false;
    }
    return true;
}

}  // namespace PhysIKA