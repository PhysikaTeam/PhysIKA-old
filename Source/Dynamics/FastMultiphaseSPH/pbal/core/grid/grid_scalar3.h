#pragma once

#include <core/macros.h>
#include <core/math/vec.h>

namespace pbal {

template <typename T>
class GridScalar3
{
private:
    std::vector<T> _data;
    Size3          _size;
    Vec3<T>        _spacing;
    Vec3<T>        _origin;

public:
    GridScalar3() {}

    GridScalar3(Size3 sz, Vec3<T> sp, Vec3<T> ori)
    {
        resize(sz, sp, ori);
    }

    ~GridScalar3()
    {
        clear();
        _data.shrink_to_fit();
    }

    void clear()
    {
        _data.clear();
    }

    void set(const T& value)
    {
        parallelForEachIndex(_data.size(),
                             [&](int idx) {
                                 _data[idx] = value;
                             });
    }

    void set(const GridScalar3& other)
    {
        _data    = other._data;
        _size    = other._size;
        _spacing = other._spacing;
        _origin  = other._origin;
    }

    void resize(Size3 sz, Vec3<T> sp, Vec3<T> ori, T initVal = T())
    {
        clear();

        _size    = sz;
        _spacing = sp;
        _origin  = ori + Vec3<T>(static_cast<T>(0.5 * sp.x), static_cast<T>(0.5 * sp.y), static_cast<T>(0.5 * sp.z));
        _data.resize(_size.x * _size.y * _size.z, initVal);
    }

    Vec3<T> dataPosition(int i, int j, int k) const
    {
        return _origin + _spacing * Vec3<T>(static_cast<T>(i), static_cast<T>(j), static_cast<T>(k));
    }

    const T& operator()(int i, int j, int k) const
    {
        return _data[i + _size.x * (j + _size.y * k)];
    }

    T& operator()(int i, int j, int k)
    {
        return _data[i + _size.x * (j + _size.y * k)];
    }

    const T& at(int i, int j, int k) const
    {
        return _data[i + _size.x * (j + _size.y * k)];
    }

    T sample(const Vec3<T>& pt) const
    {
        auto val = ConstArrayPtr3<T>(_data.data(), _size).linearSample(pt, _spacing, _origin);
        return val;
    }
};

using GridScalar3d = GridScalar3<double>;
using GridScalar3f = GridScalar3<float>;
using GridScalar3i = GridScalar3<int>;
}  // namespace pbal
