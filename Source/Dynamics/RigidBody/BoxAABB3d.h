#pragma once

#include "Framework/Framework/Node.h"

namespace PhysIKA {
template <typename T>
class BoxAABB3d
{

public:
    BoxAABB3d();
    BoxAABB3d(T lx, T ly, T lz, T ux, T uy, T uz);
    BoxAABB3d(T* pl, T* pu);

    T getlx() const
    {
        return m_l[0];
    }
    T getly() const
    {
        return m_l[1];
    }
    T getlz() const
    {
        return m_l[2];
    }
    T getux() const
    {
        return m_u[0];
    }
    T getuy() const
    {
        return m_u[1];
    }
    T getuz() const
    {
        return m_u[2];
    }

    T getl(int axis) const
    {
        return m_l[axis];
    }
    T getu(int axis) const
    {
        return m_u[axis];
    }

    bool isIntersect(const BoxAABB3d<T>& b) const;

public:
    T m_l[3];
    T m_u[3];

    //Node* m_
};

#ifdef PRECISION_FLOAT
template class BoxAABB3d<float>;
#else
template class BoxAABB3d<double>;
#endif
}  // namespace PhysIKA