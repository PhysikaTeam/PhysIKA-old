#pragma once

#include <math.h>
#include <type_traits>
#include "../../core/macros.h"

namespace pbal {

template <typename T, size_t size>
struct __vecbase
{
    // for .xyzw member
    T                      data[size];
    FLUID_CUDA_HOST_DEVICE __vecbase() {}
    template <typename... Args>
    FLUID_CUDA_HOST_DEVICE __vecbase(Args... args)
        : data{ ( T )args... }
    {
    }
};

template <typename T>
struct __vecbase<T, 2>
{
    union
    {
        T data[2];
        struct
        {
            T x, y;
        };
    };
    FLUID_CUDA_HOST_DEVICE __vecbase() {}
    FLUID_CUDA_HOST_DEVICE __vecbase(T x, T y)
        : data{ x, y } {}
};

template <typename T>
struct __vecbase<T, 3>
{
    union
    {
        T data[3];
        struct
        {
            T x, y, z;
        };
    };
    FLUID_CUDA_HOST_DEVICE __vecbase() {}
    FLUID_CUDA_HOST_DEVICE __vecbase(T x, T y, T z)
        : data{ x, y, z } {}
};

template <typename T>
struct __vecbase<T, 4>
{
    union
    {
        T data[4];
        struct
        {
            T x, y, z, w;
        };
    };
    FLUID_CUDA_HOST_DEVICE __vecbase() {}
    FLUID_CUDA_HOST_DEVICE __vecbase(T x, T y, T z, T w)
        : data{ x, y, z, w } {}
};

template <typename T>
struct cross_trait
{
    static const bool Enabled = false;
    using Operand             = struct
    {
    };
    using Result = struct
    {
    };
    FLUID_CUDA_HOST_DEVICE static Result cross(Operand a, Operand b)
    {
        return {};
    }
};

template <typename T, size_t size>
struct Vec : __vecbase<T, size>
{
    FLUID_CUDA_HOST_DEVICE Vec() {}
    template <typename... Args>
    FLUID_CUDA_HOST_DEVICE Vec(Args... args)
        : __vecbase<T, size>(args...)
    {
    }
    FLUID_CUDA_HOST_DEVICE void set(T v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] = v;
    }

    template <typename... Args>
    FLUID_CUDA_HOST_DEVICE void set(Args... args)
    {
        *this = Vec(args...);
    }

    FLUID_CUDA_HOST_DEVICE T prod()
    {
        T res = 1;
        for (int i = 0; i < size; i++)
            res *= (*this)[i];
        return res;
    }

    FLUID_CUDA_HOST_DEVICE T& operator[](size_t i)
    {
        return __vecbase<T, size>::data[i];
    }
    FLUID_CUDA_HOST_DEVICE const T& operator[](size_t i) const
    {
        return __vecbase<T, size>::data[i];
    }

    template <typename U>
    FLUID_CUDA_HOST_DEVICE Vec(const Vec<U, size>& other)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] = static_cast<T>(other[i]);
    }

    FLUID_CUDA_HOST_DEVICE Vec(const Vec& other)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] = other[i];
    }

    FLUID_CUDA_HOST_DEVICE Vec operator=(const Vec& other)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] = other[i];
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE bool isEqual(const Vec& v) const
    {
        bool result = true;
        for (size_t i = 0; i < size; i++)
            result = result && ((*this)[i] == v[i]);
        return result;
    }

    FLUID_CUDA_HOST_DEVICE T square() const
    {
        return dot(*this);
    }

    FLUID_CUDA_HOST_DEVICE T length() const
    {
        return static_cast<T>(
            sqrtf(static_cast<float>(dot(*this))));
    }

    FLUID_CUDA_HOST_DEVICE void normalize()
    {
        T l = length();
        if (l == T())
            return;
        for (size_t i = 0; i < size; i++)
            (*this)[i] /= l;
    }

    FLUID_CUDA_HOST_DEVICE Vec normalized() const
    {
        T l = length();
        if (l == T())
            return Vec();
        return this->div(l);
    }

    FLUID_CUDA_HOST_DEVICE T max() const
    {
        T result = (*this)[0];
        for (size_t i = 1; i < size; i++)
            result = MAX(result, (*this)[i]);
        return result;
    }

    FLUID_CUDA_HOST_DEVICE T min() const
    {
        T result = (*this)[0];
        for (size_t i = 1; i < size; i++)
            result = MIN(result, (*this)[i]);
        return result;
    }

    FLUID_CUDA_HOST_DEVICE T dot(const Vec& v) const
    {
        T result = T();
        for (size_t i = 0; i < size; i++)
            result += (*this)[i] * v[i];
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec add(const Vec& v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] += v[i];
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec add(T v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] += v;
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec sub(const Vec& v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] -= v[i];
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec sub(T v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] -= v;
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec mul(const Vec& v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] *= v[i];
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec mul(T v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] *= v;
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec div(const Vec& v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] /= v[i];
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec div(T v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] /= v;
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec rdiv(T v) const
    {
        Vec result = *this;
        for (size_t i = 0; i < size; i++)
        {
            result[i] = v / result[i];
        }
        return result;
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator+=(T v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] += v;
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator+=(const Vec& v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] += v[i];
        return (*this);
    }
    FLUID_CUDA_HOST_DEVICE Vec& operator-=(T v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] -= v;
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator-=(const Vec& v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] -= v[i];
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator*=(T v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] *= v;
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator*=(const Vec& v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] *= v[i];
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator/=(T v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] /= v;
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE Vec& operator/=(const Vec& v)
    {
        for (size_t i = 0; i < size; i++)
            (*this)[i] /= v[i];
        return (*this);
    }

    FLUID_CUDA_HOST_DEVICE bool operator==(const Vec& v)
    {
        return isEqual(v);
    }

    typename cross_trait<Vec>::Result cross(Vec b)
    {
        //static_assert(cross_trait<Vec>::Enabled); // c++17
        return cross_trait<Vec>::cross(*this, b);
    };
};

template <typename T>
struct cross_trait<Vec<T, 2>>
{
    static const bool Enabled = true;
    using Operand             = Vec<T, 2>;
    using Result              = T;
    FLUID_CUDA_HOST_DEVICE static Result cross(Operand a, Operand b)
    {
        return a.x * b.y - a.y * b.x;
    }
};

template <typename T>
struct cross_trait<Vec<T, 3>>
{
    static const bool Enabled = true;
    using Operand             = Vec<T, 3>;
    using Result              = Vec<T, 3>;
    FLUID_CUDA_HOST_DEVICE static Result cross(Operand a, Operand b)
    {
        return Vec<T, 3>(a.y * b.z - a.z * b.y, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
    }
};

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator+(const Vec<T, size>& a, T b)
{
    return a.add(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator+(const Vec<T, size>& a, const Vec<T, size>& b)
{
    return a.add(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator-(const Vec<T, size>& a, T b)
{
    return a.sub(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator-(const Vec<T, size>& a, const Vec<T, size>& b)
{
    return a.sub(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator*(const Vec<T, size>& a, const Vec<T, size>& b)
{
    return a.mul(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator*(const Vec<T, size>& a, T b)
{
    return a.mul(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator*(T a, const Vec<T, size>& b)
{
    return b.mul(a);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator/(const Vec<T, size>& a, const Vec<T, size>& b)
{
    return a.div(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator/(const Vec<T, size>& a, T b)
{
    return a.div(b);
}

template <typename T, size_t size>
FLUID_CUDA_HOST_DEVICE Vec<T, size> operator/(T a, const Vec<T, size>& b)
{
    return b.rdiv(a);
}

template <typename T>
using Vec2 = Vec<T, 2>;
template <typename T>
using Vec3 = Vec<T, 3>;
template <typename T>
using Vec4 = Vec<T, 4>;

using Size2 = Vec2<int>;
using Vec2d = Vec2<double>;
using Vec2f = Vec2<float>;
using Vec2i = Vec2<int>;

using Size3 = Vec3<int>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;

using Size4 = Vec4<int>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int>;
}  // namespace pbal