#pragma once
//#include "Framework/Framework/Node.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Core/Matrix/matrix_3x3.h"

#include <memory>

namespace PhysIKA {
/**
    Transformation ( in spacial vector algebra)

    A, B are frames with origins at O and P.
    rotation matrix & r = OP  are both expressed in A frame

    */

class Transformation6d
{
public:
    Transformation6d();

    template <typename T>
    Vectornd<T> transformF(const Vectornd<T>& f);

    template <typename T>
    MatrixMN<T> transformF(const MatrixMN<T>& f);

    template <typename T>
    Vectornd<T> transformM(const Vectornd<T>& m);

    template <typename T>
    MatrixMN<T> transformM(const MatrixMN<T>& m);

private:
    Matrix3f m_rotation;
    Vector3f m_translation;
};

template <typename T>
inline Vectornd<T> Transformation6d::transformF(const Vectornd<T>& f)
{
    Vectornd<T> res(6);

    for (int i = 0; i < 3; ++i)
    {
        //for
    }

    return res;
}

}  // namespace PhysIKA