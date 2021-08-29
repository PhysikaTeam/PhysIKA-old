#pragma once

#include <core/array/array_ptr1.h>
#include <core/array/array_ptr2.h>
#include <core/macros.h>
#include <core/math/vec.h>
#include <core/utils/parallel.h>

namespace pbal {

template <typename T>
class GridScalar2
{
private:
    std::vector<T> _data;
    Size2          _size;
    Vec2<T>        _spacing;
    Vec2<T>        _origin;

public:
    GridScalar2() {}

    GridScalar2(Size2 sz, Vec2<T> sp, Vec2<T> ori)
    {
        resize(sz, sp, ori);
    }

    ~GridScalar2()
    {
        clear();
        _data.shrink_to_fit();
    }

    void set(const T& value)
    {
        parallelForEachIndex(_data.size(),
                             [&](int idx) {
                                 _data[idx] = value;
                             });
    }

    void set(const GridScalar2& other)
    {
        _data    = other._data;
        _size    = other._size;
        _spacing = other._spacing;
        _origin  = other._origin;
    }

    void clear()
    {
        _data.clear();
    }

    void resize(Size2 sz, Vec2<T> sp, Vec2<T> ori, T initVal = T())
    {
        clear();

        _size    = sz;
        _spacing = sp;
        _origin  = ori + Vec2<T>(static_cast<T>(0.5 * sp.x), static_cast<T>(0.5 * sp.y));
        _data.resize(_size.x * _size.y, initVal);
    }

    const Size2& size() const
    {
        return _size;
    }

    ArrayPtr2<T> dataAccessor()
    {
        return ArrayPtr2<T>(
            reinterpret_cast<T*>(&_data[0]),
            _size);
    }

    Vec2<T> dataPosition(int i, int j) const
    {
        return _origin + _spacing * Vec2<T>(static_cast<T>(i), static_cast<T>(j));
    }

    const T& operator()(int i, int j) const
    {
        return _data[i + j * _size.x];
    }

    T& operator()(int i, int j)
    {
        return _data[i + j * _size.x];
    }

    T sample(const Vec2<T>& pt) const
    {
        auto val = ConstArrayPtr2<T>(_data.data(), _size).linearSample(pt, _spacing, _origin);
        return val;
    }
};

using GridScalar2d = GridScalar2<double>;
using GridScalar2f = GridScalar2<float>;
using GridScalar2i = GridScalar2<int>;

}  // namespace pbal
