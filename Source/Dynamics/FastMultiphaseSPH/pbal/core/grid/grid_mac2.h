#pragma once

#include <core/math/vec.h>
#include <core/array/array_ptr2.h>
#include <core/array/const_array_ptr2.h>
namespace pbal {

template <typename T>
class GridMac2
{
private:
    std::vector<T> _u, _v;
    Size2          _uSize, _vSize;
    Vec2<T>        _spacing;
    Vec2<T>        _uOrigin, _vOrigin;

public:
    GridMac2() {}

    GridMac2(Size2 sz, Vec2<T> sp, Vec2<T> ori)
    {
        resize(sz, sp, ori);
    }

    ~GridMac2()
    {
        clear();
        _u.shrink_to_fit();
        _v.shrink_to_fit();
    }

    void set(const T& value)
    {
        std::fill(_u.begin(), _u.end(), value);
        std::fill(_v.begin(), _v.end(), value);
    }

    void set(const GridMac2& other)
    {
        _u       = other._u;
        _v       = other._v;
        _uSize   = other._uSize;
        _vSize   = other._vSize;
        _spacing = other._spacing;
        _uOrigin = other._uOrigin;
        _vOrigin = other._vOrigin;
    }

    void clear()
    {
        _u.clear();
        _v.clear();
    }

    void resize(Size2 sz, Vec2<T> sp, Vec2<T> ori, T initVal = T())
    {
        clear();

        _spacing = sp;
        _uOrigin = ori + Vec2<T>(static_cast<T>(0.0), static_cast<T>(0.5 * _spacing.y));
        _vOrigin = ori + Vec2<T>(static_cast<T>(0.5 * _spacing.x), static_cast<T>(0.0));

        auto _size = sz;
        _uSize     = Size2(_size.x + 1, _size.y);
        _vSize     = Size2(_size.x, _size.y + 1);

        _u.resize(_uSize.x * _uSize.y, initVal);
        _v.resize(_vSize.x * _vSize.y, initVal);
    }

    Size2 uSize() const
    {
        return _uSize;
    }

    Size2 vSize() const
    {
        return _vSize;
    }

    Vec2<T> uOrigin() const
    {
        return _uOrigin;
    }

    Vec2<T> vOrigin() const
    {
        return _vOrigin;
    }

    ArrayPtr2<T> uAccessor()
    {
        return ArrayPtr2<T>(
            reinterpret_cast<T*>(&_u[0]),
            _uSize);
    }

    ConstArrayPtr2<T> uConstAccessor() const
    {
        return ConstArrayPtr2<T>(
            _u.data(),
            _uSize);
    }

    ArrayPtr2<T> vAccessor()
    {
        return ArrayPtr2<T>(
            reinterpret_cast<T*>(&_v[0]),
            _vSize);
    }

    ConstArrayPtr2<T> vConstAccessor() const
    {
        return ConstArrayPtr2<T>(
            _v.data(),
            _vSize);
    }

    Vec2<T> uPosition(int i, int j) const
    {
        return _uOrigin + _spacing * Vec2<T>(static_cast<T>(i), static_cast<T>(j));
    }

    Vec2<T> vPosition(int i, int j) const
    {
        return _vOrigin + _spacing * Vec2<T>(static_cast<T>(i), static_cast<T>(j));
    }

    Vec2<T> sample(const Vec2<T>& pt) const
    {
        auto uVal = uConstAccessor().linearSample(
            pt,
            _spacing,
            _uOrigin);
        auto vVal = vConstAccessor().linearSample(
            pt,
            _spacing,
            _vOrigin);
        return Vec2<T>(uVal, vVal);
    }
};

using GridMac2d = GridMac2<double>;
using GridMac2f = GridMac2<float>;
using GridMac2i = GridMac2<int>;

}  // namespace pbal
