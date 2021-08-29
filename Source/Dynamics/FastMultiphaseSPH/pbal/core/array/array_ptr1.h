#pragma once

#include <core/macros.h>

namespace pbal {

template <typename T>
class ArrayPtr1
{
public:
    ArrayPtr1() {}

    ArrayPtr1(T* data, int size)
        : _data(data), _size(size) {}

    int size() const
    {
        return _size;
    }

    T& operator()(int i)
    {
        return _data[i];
    }

    T operator()(int i) const
    {
        return _data[i];
    }

    T& operator[](int i)
    {
        return _data[i];
    }

    T operator[](int i) const
    {
        return _data[i];
    }

private:
    T*  _data = nullptr;
    int _size = 0;
};

}  // namespace pbal