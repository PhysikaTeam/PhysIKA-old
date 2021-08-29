#pragma once
#include <cmath>
#include "Vec.h"

namespace gui {

template <typename T>
class Quaternion
{
public:
    T x, y, z, w;
    Quaternion()
    {
    }

    Quaternion(T xx, T yy, T zz, T ww)
    {
        x = xx;
        y = yy;
        z = zz;
        w = ww;
    }

    Quaternion(T rot, const Vec<3, T>& axis)
    {
        T c = cos(0.5f * rot);
        T s = sin(0.5f * rot);
        T t = s / axis.Length();
        x   = c;
        y   = axis.x * t;
        z   = axis.y * t;
        w   = axis.z * t;
    }

    void Rotate(Vec<3, T>& xx)
    {
        T xlen = xx.Length();
        if (xlen == 0.0f)
            return;

        Quaternion p(0, xx.x, xx.y, xx.z);
        Quaternion qbar(x, -y, -z, -w);
        Quaternion qtmp;
        qtmp = ComposeWith(p);
        qtmp = qtmp.ComposeWith(qbar);
        xx.x = qtmp.y;
        xx.y = qtmp.z;
        xx.z = qtmp.w;
        xx.Normalize();
        xx *= xlen;
    }

    Quaternion ComposeWith(const Quaternion& q)
    {
        Quaternion result;
        result.x = x * q.x - y * q.y - z * q.z - w * q.w;
        result.y = x * q.y + y * q.x + z * q.w - w * q.z;
        result.z = x * q.z + z * q.x + w * q.y - y * q.w;
        result.w = x * q.w + w * q.x + y * q.z - z * q.y;
        result.Normalize();
        return result;
    }

    void Normalize()
    {
        T d = sqrt(x * x + y * y + z * z + w * w);
        if (d == 0)
        {
            x = 1.0f;
            y = z = w = 0.0f;
            return;
        }
        d = 1.0f / d;
        x *= d;
        y *= d;
        z *= d;
        w *= d;
    }

    void ToRotAxis(T& rot, Vec<3, T>& axis)
    {
        rot = 2.0f * acos(x);
        if (rot == 0)
        {
            axis.x = axis.y = 0;
            axis.z          = 1;
            return;
        }
        axis.x = y;
        axis.y = z;
        axis.z = w;
        axis.Normalize();
    }
};

template class Quaternion<float>;
template class Quaternion<double>;

typedef Quaternion<float>  Quat1f;
typedef Quaternion<double> Quat1d;

}  // namespace gui