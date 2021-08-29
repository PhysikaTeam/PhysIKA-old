#pragma once

#include <core/macros.h>
#include <core/utils/host_device_utils.h>
#include <core/math/vec.h>

namespace pbal {

template <typename T>
class ConstArrayPtr3
{
public:
    ConstArrayPtr3(const T* data, Size3 size)
        : _data(data), _size(size) {}

    Size3 size() const
    {
        return _size;
    }

    const T& at(int i, int j, int k) const
    {
        return _data[i + _size.x * (j + _size.y * k)];
    }

    const T& operator()(int i, int j, int k) const
    {
        return _data[i + _size.x * (j + _size.y * k)];
    }

    const T& operator[](int i) const
    {
        return _data[i];
    }

    T max() const
    {
        T m = T();
        for (int i = 0; i < _size.x * _size.y; i++)
        {
            if (_data[i] > m)
            {
                m = _data[i];
            }
        }
        return m;
    }

    T linearSample(
        Vec3<T> pt,
        Vec3<T> spacing,
        Vec3<T> origin) const
    {

        int     i, j, k;
        T       fx, fy, fz;
        Vec3<T> pt0 = (pt - origin) / spacing;

        getBarycentric(pt0.x, 0, _size.x - 1, &i, &fx);
        getBarycentric(pt0.y, 0, _size.y - 1, &j, &fy);
        getBarycentric(pt0.z, 0, _size.z - 1, &k, &fz);

        int ip1 = MIN(i + 1, _size.x - 1);
        int jp1 = MIN(j + 1, _size.y - 1);
        int kp1 = MIN(k + 1, _size.z - 1);

        return trilerp(
            at(i, j, k),
            at(ip1, j, k),
            at(i, jp1, k),
            at(ip1, jp1, k),
            at(i, j, kp1),
            at(ip1, j, kp1),
            at(i, jp1, kp1),
            at(ip1, jp1, kp1),
            fx,
            fy,
            fz);
    }

private:
    const T* _data = nullptr;
    Size3    _size;
};

using ConstArrayPtr3d = ConstArrayPtr3<double>;
using ConstArrayPtr3f = ConstArrayPtr3<float>;
using ConstArrayPtr3i = ConstArrayPtr3<int>;

}  // namespace pbal
