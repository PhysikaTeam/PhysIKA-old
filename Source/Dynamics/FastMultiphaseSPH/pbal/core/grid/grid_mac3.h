#pragma once

#include <core/array/array_ptr3.h>
#include <core/array/const_array_ptr3.h>
#include <core/math/vec.h>

namespace pbal {

template <typename T>
class GridMac3
{
public:
    std::vector<T> _u, _v, _w;
    Size3          _uSize, _vSize, _wSize;
    Vec3<T>        _spacing;
    Vec3<T>        _uOrigin, _vOrigin, _wOrigin;

    GridMac3() {}

    GridMac3(Size3 sz, Vec3<T> sp, Vec3<T> ori)
    {
        resize(sz, sp, ori);
    }

    ~GridMac3()
    {
        clear();
        _u.shrink_to_fit();
        _v.shrink_to_fit();
        _w.shrink_to_fit();
    }

    void set(const T& value)
    {
        std::fill(_u.begin(), _u.end(), value);
        std::fill(_v.begin(), _v.end(), value);
        std::fill(_w.begin(), _w.end(), value);
    }

    void set(const GridMac3& other)
    {
        _u       = other._u;
        _v       = other._v;
        _w       = other._w;
        _uSize   = other._uSize;
        _vSize   = other._vSize;
        _wSize   = other._wSize;
        _spacing = other._spacing;
        _uOrigin = other._uOrigin;
        _vOrigin = other._vOrigin;
        _wOrigin = other._wOrigin;
    }

    void clear()
    {
        _u.clear();
        _v.clear();
        _w.clear();
    }

    void resize(Size3 size, Vec3<T> spacing, Vec3<T> origin, T initVal = T())
    {
        clear();

        _spacing = spacing;
        _uOrigin = origin + Vec3<T>(static_cast<T>(0.0), static_cast<T>(0.5 * _spacing.y), static_cast<T>(0.5 * _spacing.z));
        _vOrigin = origin + Vec3<T>(static_cast<T>(0.5 * _spacing.x), static_cast<T>(0.0), static_cast<T>(0.5 * _spacing.z));
        _wOrigin = origin + Vec3<T>(static_cast<T>(0.5 * _spacing.x), static_cast<T>(0.5 * _spacing.y), static_cast<T>(0.0));

        _uSize = Size3(size.x + 1, size.y, size.z);
        _vSize = Size3(size.x, size.y + 1, size.z);
        _wSize = Size3(size.x, size.y, size.z + 1);

        _u.resize(_uSize.x * _uSize.y * _uSize.z, initVal);
        _v.resize(_vSize.x * _vSize.y * _vSize.z, initVal);
        _w.resize(_wSize.x * _wSize.y * _wSize.z, initVal);
    }

    Size3 uSize() const
    {
        return _uSize;
    }

    Size3 vSize() const
    {
        return _vSize;
    }

    Size3 wSize() const
    {
        return _wSize;
    }

    Vec3<T> uOrigin() const
    {
        return _uOrigin;
    }

    Vec3<T> vOrigin() const
    {
        return _vOrigin;
    }

    Vec3<T> wOrigin() const
    {
        return _wOrigin;
    }

    ArrayPtr3<T> uAccessor()
    {
        return ArrayPtr3<T>(
            reinterpret_cast<T*>(&_u[0]),
            _uSize);
    }

    ConstArrayPtr3<T> uConstAccessor() const
    {
        return ConstArrayPtr3<T>(
            _u.data(),
            _uSize);
    }

    ArrayPtr3<T> vAccessor()
    {
        return ArrayPtr3<T>(
            reinterpret_cast<T*>(&_v[0]),
            _vSize);
    }

    ConstArrayPtr3<T> vConstAccessor() const
    {
        return ConstArrayPtr3<T>(
            _v.data(),
            _vSize);
    }

    ArrayPtr3<T> wAccessor()
    {
        return ArrayPtr3<T>(
            reinterpret_cast<T*>(&_w[0]),
            _wSize);
    }

    ConstArrayPtr3<T> wConstAccessor() const
    {
        return ConstArrayPtr3<T>(
            _w.data(),
            _wSize);
    }

    Vec3<T> uPosition(int i, int j, int k) const
    {
        return _uOrigin + _spacing * Vec3<T>(static_cast<T>(i), static_cast<T>(j), static_cast<T>(k));
    }

    Vec3<T> vPosition(int i, int j, int k) const
    {
        return _vOrigin + _spacing * Vec3<T>(static_cast<T>(i), static_cast<T>(j), static_cast<T>(k));
    }

    Vec3<T> wPosition(int i, int j, int k) const
    {
        return _wOrigin + _spacing * Vec3<T>(static_cast<T>(i), static_cast<T>(j), static_cast<T>(k));
    }

    Vec3<T> sample(const Vec3<T>& pt) const
    {
        auto uVal = ConstArrayPtr3<T>(_u.data(), _uSize).linearSample(pt, _spacing, _uOrigin);
        auto vVal = ConstArrayPtr3<T>(_v.data(), _vSize).linearSample(pt, _spacing, _vOrigin);
        auto wVal = ConstArrayPtr3<T>(_w.data(), _wSize).linearSample(pt, _spacing, _wOrigin);
        return Vec3<T>(uVal, vVal, wVal);
    }
};

using GridMac3d = GridMac3<double>;
using GridMac3f = GridMac3<float>;
using GridMac3i = GridMac3<int>;

}  // namespace pbal
