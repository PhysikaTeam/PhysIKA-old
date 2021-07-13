#pragma once

#include <core/macros.h>
#include <core/utils/host_device_utils.h>
#include <core/math/vec.h>

namespace pbal {

template <typename T>
class ArrayPtr2
{
public:
    ArrayPtr2(T* data, Size2 size)
        : _data(data), _size(size) {}

    Size2 size() const
    {
        return _size;
    }

    T& at(int i, int j)
    {
        return _data[i + j * _size.x];
    }

    const T& at(int i, int j) const
    {
        return _data[i + j * _size.x];
    }

    T* addressAt(int i, int j)
    {
        return &_data[i + j * _size.x];
    }

    T& operator()(int i, int j)
    {
        return _data[i + j * _size.x];
    }

    const T& operator()(int i, int j) const
    {
        return _data[i + j * _size.x];
    }

    T& operator[](int i)
    {
        return _data[i];
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
        Vec2<T> pt,
        Vec2<T> spacing,
        Vec2<T> origin) const
    {

        int     i, j;
        T       fx, fy;
        Vec2<T> pt0 = (pt - origin) / spacing;

        getBarycentric(pt0.x, 0, _size.x - 1, &i, &fx);
        getBarycentric(pt0.y, 0, _size.y - 1, &j, &fy);

        int ip1 = MIN(i + 1, _size.x - 1);
        int jp1 = MIN(j + 1, _size.y - 1);

        return bilerp(
            at(i, j),
            at(ip1, j),
            at(i, jp1),
            at(ip1, jp1),
            fx,
            fy);
    }

private:
    T*    _data = nullptr;
    Size2 _size;
};

using ArrayPtr2d = ArrayPtr2<double>;
using ArrayPtr2f = ArrayPtr2<float>;
using ArrayPtr2i = ArrayPtr2<int>;

}  // namespace pbal
