#pragma once
#include <cmath>
#include <algorithm>
#include "StaticAssert.h"

using namespace std;
namespace gui {

#define EPSILON 1e-6

/*!
 *    \struct    Vec
 *    \brief    Template for 2D or 3D vector with components of any number type.
 */
template <int dim, typename T>
class Vec;
template <int dim, typename T>
Vec<dim, T> operator*(T d, const Vec<dim, T>& v);

template <int dim, typename T>
class Vec
{
public:
    enum PARAMETER
    {
        m = dim
    };
    STATIC_ASSERT(m > 4);

    T x[m];

    Vec()
    {
        for (int i = 0; i < m; i++)
            x[i] = 0;
    }

    Vec(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] = d;
    }

    Vec(Vec& v)
    {
        for (int i = 0; i < m; i++)
            x[i] = v.x[i];
    }

    inline T& operator[](unsigned int id)
    {
        return x[id];
    }

    inline T operator[](unsigned int id) const
    {
        return x[id];
    }

    inline T Dot(const Vec& v)
    {
        T d = 0;
        for (int i = 0; i < m; i++)
            d += x[i] * v.x[i];
        return d;
    }

    inline T LengthSq()
    {
        T d = 0;
        for (int i = 0; i < m; i++)
            d += x[i] * x[i];
        return d;
    }

    inline T Length()
    {
        return sqrt(LengthSq());
    }

    inline const Vec operator-(void) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = -x[i];
        return result;
    }

    inline Vec operator+(const Vec& v) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] + v.x[i];
        return result;
    }

    inline Vec operator-(const Vec& v) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] - v.x[i];
        return result;
    }

    inline Vec operator+(T d) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] + d;
        return result;
    }

    inline Vec operator-(T d) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] - d;
        return result;
    }

    inline Vec operator*(T d) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] * d;
        return result;
    }

    inline Vec operator/(T d) const
    {
        Vec result;
        for (int i = 0; i < m; i++)
            result[i] = x[i] / d;
        return result;
    }

    inline void operator+=(const Vec& v)
    {
        for (int i = 0; i < m; i++)
            x[i] += v.x[i];
    }

    inline void operator-=(const Vec& v)
    {
        for (int i = 0; i < m; i++)
            x[i] -= v.x[i];
    }

    inline void operator+=(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] += d;
    }

    inline void operator-=(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] -= d;
    }

    inline void operator*=(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] *= d;
    }

    inline void operator/=(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] /= d;
    }

    inline void operator=(T d)
    {
        for (int i = 0; i < m; i++)
            x[i] = d;
    }

    friend Vec<dim, T> operator*(T d, const Vec<dim, T>& v)
    {
        return v * d;
    }
};

/*!
 *    \brief    2D vector with components of any number type.
 */
template <typename T>
class Vec<2, T>
{
public:
    T x, y;

    Vec()
    {
        x = y = 0;
    }

    Vec(T v)
    {
        x = y = v;
    }

    Vec(T X, T Y)
    {
        x = X;
        y = Y;
    }

    Vec(T X, T Y, T Z)
    {
        x = X;
        y = Y;
    }

    Vec(T* v)
    {
        x = v[0];
        y = v[1];
    }

    template <int dim_, typename T_>
    Vec<2, T>& operator=(const Vec<dim_, T_>& other)
    {
        x = ( T )other.x;
        y = ( T )other.y;
        return *this;
    }

    template <int dim_, typename T_>
    operator Vec<dim_, T_>()
    {
        return Vec<dim_, T_>(static_cast<T_>(x), static_cast<T_>(y));
    }

    inline T& operator[](unsigned int id)
    {
        return id ? y : x;  //return (&x)[id];
    }

    inline T operator[](unsigned int id) const
    {
        return id ? y : x;  //return (&x)[id];
    }

    inline T Dot(const Vec& v) const
    {
        return x * v.x + y * v.y;
    }

    inline T LengthSq() const
    {
        return x * x + y * y;
    }

    inline T Length() const
    {
        return sqrt(LengthSq());
    }

    inline T Normalize()
    {
        T l = Length();
        if (l < EPSILON)
        {
            x = 0.0f;
            y = 0.0f;
        }
        else
        {
            x /= l;
            y /= l;
        }
        return l;
    }

    inline Vec Rotate(T angle) const
    {
        return Vec(x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle));
    }

    inline const Vec operator-(void) const
    {
        return Vec(-x, -y);
    }

    inline Vec operator+(const Vec& v) const
    {
        return Vec(x + v.x, y + v.y);
    }

    inline Vec operator-(const Vec& v) const
    {
        return Vec(x - v.x, y - v.y);
    }

    inline Vec operator+(T d) const
    {
        return Vec(x + d, y + d);
    }

    inline Vec operator-(T d) const
    {
        return Vec(x - d, y - d);
    }

    inline Vec operator*(T d) const
    {
        return Vec(x * d, y * d);
    }

    inline Vec operator*(const Vec& v) const
    {
        return Vec(x * v.x, y * v.y);
    }

    inline Vec operator/(T d) const
    {
        return Vec(x / d, y / d);
    }

    inline Vec operator/(const Vec& v) const
    {
        return Vec(x / v.x, y / v.y);
    }

    inline void operator+=(const Vec& v)
    {
        x += v.x;
        y += v.y;
    }

    inline void operator-=(const Vec& v)
    {
        x -= v.x;
        y -= v.y;
    }

    inline void operator+=(T d)
    {
        x += d;
        y += d;
    }

    inline void operator-=(T d)
    {
        x -= d;
        y -= d;
    }

    inline void operator*=(T d)
    {
        x *= d;
        y *= d;
    }

    inline void operator/=(T d)
    {
        x /= d;
        y /= d;
    }

    inline void operator=(T d)
    {
        x = d;
        y = d;
    }

    friend Vec<2, T> operator*(T d, const Vec<2, T>& v)
    {
        return v * d;
    }
};

/*!
 *    \brief    3D vector with components of any number type.
 */
template <typename T>
class Vec<3, T>
{
public:
    T x, y, z;

    Vec()
    {
        x = y = z = 0;
    }

    Vec(T v)
    {
        x = y = z = v;
    }

    Vec(T X, T Y)
    {
        x = X;
        y = Y;
        z = 0;
    }

    Vec(T X, T Y, T Z)
    {
        x = X;
        y = Y;
        z = Z;
    }

    Vec(T* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    template <typename T_>
    Vec<3, T>& operator=(const Vec<3, T_>& other)
    {
        x = ( T )other.x;
        y = ( T )other.y;
        z = ( T )other.z;
        return *this;
    }

    template <int dim_, typename T_>
    operator Vec<dim_, T_>()
    {
        return Vec<dim_, T_>(static_cast<T_>(x), static_cast<T_>(y), static_cast<T_>(z));
    }

    inline T& operator[](unsigned int id)
    {
        return (&x)[id];
    }

    inline T operator[](unsigned int id) const
    {
        return (&x)[id];
    }

    inline T Dot(const Vec& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    inline Vec Cross(const Vec& v) const
    {
        return Vec(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    inline T Normalize()
    {
        T l = Length();
        if (l < EPSILON)
        {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }
        else
        {
            x /= l;
            y /= l;
            z /= l;
        }
        return l;
    }

    inline T LengthSq() const
    {
        return x * x + y * y + z * z;
    }

    inline T Length() const
    {
        return sqrt(LengthSq());
    }

    inline const Vec operator-(void) const
    {
        return Vec(-x, -y, -z);
    }

    inline Vec operator+(const Vec& v) const
    {
        return Vec(x + v.x, y + v.y, z + v.z);
    }

    inline Vec operator-(const Vec& v) const
    {
        return Vec(x - v.x, y - v.y, z - v.z);
    }

    inline Vec operator+(T d) const
    {
        return Vec(x + d, y + d, z + d);
    }

    inline Vec operator-(T d) const
    {
        return Vec(x - d, y - d, z - d);
    }

    inline Vec operator*(T d) const
    {
        return Vec(x * d, y * d, z * d);
    }

    inline Vec operator*(const Vec& v) const
    {
        return Vec(x * v.x, y * v.y, z * v.z);
    }

    inline Vec operator/(T d) const
    {
        return Vec(x / d, y / d, z / d);
    }

    inline Vec operator/(const Vec& v) const
    {
        return Vec(x / v.x, y / v.y, z / v.z);
    }

    inline void operator+=(const Vec& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    inline void operator-=(const Vec& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    inline void operator+=(T d)
    {
        x += d;
        y += d;
        z += d;
    }

    inline void operator-=(T d)
    {
        x -= d;
        y -= d;
        z -= d;
    }

    inline void operator*=(T d)
    {
        x *= d;
        y *= d;
        z *= d;
    }

    inline void operator/=(T d)
    {
        x /= d;
        y /= d;
        z /= d;
    }

    inline void operator=(T d)
    {
        x = d;
        y = d;
        z = d;
    }

    friend Vec<3, T> operator*(T d, const Vec<3, T>& v)
    {
        return v * d;
    }
};

/*!
 *    \brief    4D vector with components of any number type.
 */
template <typename T>
class Vec<4, T>
{
public:
    T x, y, z, w;

    Vec()
    {
        x = y = z = w = 0;
    }

    Vec(T v)
    {
        x = y = z = w = v;
    }

    Vec(T X, T Y, T Z)
    {
        x = X;
        y = Y;
        z = Z;
        w = 0;
    }

    Vec(T X, T Y, T Z, T W)
    {
        x = X;
        y = Y;
        z = Z;
        w = W;
    }

    Vec(T* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
        w = v[3];
    }

    inline T& operator[](unsigned int id)
    {
        return (&x)[id];
    }

    inline T operator[](unsigned int id) const
    {
        return (&x)[id];
    }

    inline T Dot(const Vec& v)
    {
        return x * v.x + y * v.y + z * v.z - w * v.w;
    }

    inline T LengthSq()
    {
        return x * x + y * y + z * z + w * w;
    }

    inline T Length()
    {
        return sqrt(LengthSq());
    }

    inline T Normalize()
    {
        T l = Length();
        if (l < EPSILON)
        {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 0.0f;
        }
        else
        {
            x /= l;
            y /= l;
            z /= l;
            w /= l;
        }
        return l;
    }

    inline const Vec operator-(void) const
    {
        return Vec(-x, -y, -z, -w);
    }

    inline Vec operator+(const Vec& v) const
    {
        return Vec(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    inline Vec operator-(const Vec& v) const
    {
        return Vec(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    inline Vec operator+(T d) const
    {
        return Vec(x + d, y + d, z + d, w + d);
    }

    inline Vec operator-(T d) const
    {
        return Vec(x - d, y - d, z - d, w - d);
    }

    inline Vec operator*(T d) const
    {
        return Vec(x * d, y * d, z * d, w * d);
    }

    inline Vec operator*(const Vec& v) const
    {
        return Vec(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    inline Vec operator/(T d) const
    {
        return Vec(x / d, y / d, z / d, w / d);
    }

    inline Vec operator/(const Vec& v) const
    {
        return Vec(x / v.x, y / v.y, z / v.z, w / v.w);
    }

    inline void operator+=(const Vec& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
    }

    inline void operator-=(const Vec& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
    }

    inline void operator+=(T d)
    {
        x += d;
        y += d;
        z += d;
        w += d;
    }

    inline void operator-=(T d)
    {
        x -= d;
        y -= d;
        z -= d;
        w -= d;
    }

    inline void operator*=(T d)
    {
        x *= d;
        y *= d;
        z *= d;
        w *= d;
    }

    inline void operator/=(T d)
    {
        x /= d;
        y /= d;
        z /= d;
        w /= d;
    }

    inline void operator=(T d)
    {
        x = d;
        y = d;
        z = d;
        w = d;
    }

    friend Vec<4, T> operator*(T d, const Vec<4, T>& v)
    {
        return v * d;
    }
};

/*!
 *    \brief    Multiplication operator: number*vector (number is left operand).
 */
// template<int dim, typename T, typename typ2>
// inline Vec<dim,T> operator * (typ2 d, const Vec<dim,T>& v)
// {
//     return v * static_cast<T>(d);
// }

// some easy vector creating

template <typename typ1, typename typ2>
inline Vec<2, float> Vec2f(typ1 x, typ2 y)
{
    return Vec<2, float>(( float )x, ( float )y);
}

template <typename typ1, typename typ2>
inline Vec<2, double> Vec2d(typ1 x, typ2 y)
{
    return Vec<2, double>(x, y);
}

template <typename typ1, typename typ2, typename typ3>
inline Vec<3, float> Vec3f(typ1 x, typ2 y, typ3 z)
{
    return Vec<3, float>(( float )x, ( float )y, ( float )z);
}

template <typename typ1, typename typ2, typename typ3>
inline Vec<3, double> Vec3d(typ1 x, typ2 y, typ3 z)
{
    return Vec<3, double>(x, y, z);
}

template <typename T>
inline T DistanceSq(const Vec<2, T>& v1, const Vec<2, T>& v2)
{
    return Vec<2, T>(v1.x - v2.x, v1.y - v2.y).LengthSq();
}

template <typename T>
inline T DistanceSq(const Vec<3, T>& v1, const Vec<3, T>& v2)
{
    return Vec<3, T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z).LengthSq();
}

template <typename T>
inline T DistanceSq(const Vec<4, T>& v1, const Vec<4, T>& v2)
{
    return Vec<4, T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w).LengthSq();
}

template <typename T>
inline T Distance(const Vec<2, T>& v1, const Vec<2, T>& v2)
{
    return Vec<2, T>(v1.x - v2.x, v1.y - v2.y).Length();
}

template <typename T>
inline T Distance(const Vec<3, T>& v1, const Vec<3, T>& v2)
{
    return Vec<3, T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z).Length();
}

template <typename T>
inline T Distance(const Vec<4, T>& v1, const Vec<4, T>& v2)
{
    return Vec<4, T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w).Length();
}

template <typename T>
inline Vec<2, T> Max(const Vec<2, T>& v1, const Vec<2, T>& v2)
{
    return Vec<2, T>(max(v1.x, v2.x), max(v1.y, v2.y));
}

template <typename T>
inline Vec<3, T> Max(const Vec<3, T>& v1, const Vec<3, T>& v2)
{
    return Vec<3, T>(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z));
}

template <typename T>
inline Vec<4, T> Max(const Vec<4, T>& v1, const Vec<4, T>& v2)
{
    return Vec<4, T>(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z), max(v1.w, v2.w));
}

template <typename T>
inline Vec<2, T> Min(const Vec<2, T>& v1, const Vec<2, T>& v2)
{
    return Vec<2, T>(min(v1.x, v2.x), min(v1.y, v2.y));
}

template <typename T>
inline Vec<3, T> Min(const Vec<3, T>& v1, const Vec<3, T>& v2)
{
    return Vec<3, T>(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z));
}

template <typename T>
inline Vec<4, T> Min(const Vec<4, T>& v1, const Vec<4, T>& v2)
{
    return Vec<4, T>(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z), min(v1.w, v2.w));
}

template class Vec<2, float>;
template class Vec<2, double>;
template class Vec<3, float>;
template class Vec<3, double>;
template class Vec<4, float>;
template class Vec<4, double>;

typedef Vec<2, float>  Vector2f;
typedef Vec<2, double> Vector2d;
typedef Vec<3, float>  Vector3f;
typedef Vec<3, double> Vector3d;
typedef Vec<4, float>  Vector4f;
typedef Vec<4, double> Vector4d;

}  // namespace gui
