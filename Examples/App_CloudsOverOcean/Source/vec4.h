#ifndef _VEC4_H_
#define _VEC4_H_

#include <cassert>

template <typename type> class vec4
{

public:
    type x, y, z, w;

    vec4();

    vec4(type xi, type yi, type zi, type wi);

    vec4(const type v[4]);

    vec4(const vec4& v);

    type operator[](const int i) const;

    type& operator[](const int i);

    vec4 operator=(const vec4& v);

    bool operator==(const vec4& v) const;

    bool operator!=(const vec4& v) const;

    vec4 operator+(const vec4& v) const;

    vec4 operator-(const vec4& v) const;

    vec4 operator*(const vec4& v) const;

    vec4 operator*(const type scalar) const;

    vec4 operator/(const vec4& v) const;

    vec4 operator/(const type scalar) const;

    vec4 operator-() const;

    vec4& operator+=(const vec4& v);

    vec4& operator-=(const vec4& v);

    vec4& operator*=(const type& scalar);

    vec4& operator/=(const type& scalar);

    type dotproduct(const vec4& v) const;

    template <class t>
    vec4<t> cast() {
        return vec4<t>((t) x, (t) y, (t) z, (t )w);
    }

    static const vec4 ZERO;

    static const vec4 UNIT_X;

    static const vec4 UNIT_Y;

    static const vec4 UNIT_Z;

    static const vec4 UNIT_W;
};

typedef vec4<float> vec4f;

typedef vec4<double> vec4d;

typedef vec4<int> vec4i;

template <typename type>
inline vec4<type>::vec4()
{
}

template <typename type>
inline vec4<type>::vec4(type xi, type yi, type zi, type wi) :
    x(xi), y(yi), z(zi), w(wi)
{
}

template <typename type>
inline vec4<type>::vec4(const type v[4]) :
    x(v[0]), y(v[1]), z(v[2]), w(v[3])
{
}

template <typename type>
inline vec4<type>::vec4(const vec4& v) :
    x(v.x), y(v.y), z(v.z), w(v.w)
{
}

template <typename type>
inline type vec4<type>::operator[](const int i) const
{
    return *(&x + i);
}

template <typename type>
inline type& vec4<type>::operator[](const int i)
{
    return *(&x + i);
}

template <typename type>
inline vec4<type> vec4<type>::operator=(const vec4<type>& v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
    return *this;
}

template <typename type>
inline bool vec4<type>::operator==(const vec4<type>& v) const
{
    return (x == v.x && y == v.y && z == v.z && w == v.w);
}

template <typename type>
inline bool vec4<type>::operator!=(const vec4<type>& v) const
{
    return (x != v.x || y != v.y || z != v.z || w != v.w);
}

template <typename type>
inline vec4<type> vec4<type>::operator+(const vec4<type>& v) const
{
    return vec4(x + v.x, y + v.y, z + v.z, w + v.w);
}

template <typename type>
inline vec4<type> vec4<type>::operator-(const vec4<type>& v) const
{
    return vec4(x - v.x, y - v.y, z - v.z, w - v.w);
}

template <typename type>
inline vec4<type> vec4<type>::operator*(const vec4<type>& v) const
{
    return vec4(x * v.x, y * v.y, z * v.z, w * v.w);
}

template <typename type>
inline vec4<type> vec4<type>::operator*(const type scalar) const
{
    return vec4(x * scalar, y * scalar, z * scalar, w * scalar);
}

template <typename type>
inline vec4<type> vec4<type>::operator/(const vec4<type>& v) const
{
    return vec4(x / v.x, y / v.y, z / v.z, w / v.w);
}

template <typename type>
inline vec4<type> vec4<type>::operator/(const type scalar) const
{
    assert(scalar != 0);
    type inv = 1 / scalar;
    return vec4(x * inv, y * inv, z * inv, w * inv);
}

template <typename type>
inline vec4<type> vec4<type>::operator-() const
{
    return vec4(-x, -y, -z, -w);
}

template <typename type>
inline vec4<type>& vec4<type>::operator+=(const vec4<type>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

template <typename type>
inline vec4<type>& vec4<type>::operator-=(const vec4<type>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

template <typename type>
inline vec4<type>& vec4<type>::operator*=(const type& scalar)
{
    x *= scalar;
    y *= scalar;
    z *= scalar;
    w *= scalar;
    return *this;
}

template <typename type>
inline vec4<type>& vec4<type>::operator/=(const type& scalar)
{
    assert(scalar != 0);
    type inv = 1 / scalar;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
    return *this;
}

template <typename type>
inline type vec4<type>::dotproduct(const vec4<type>& v) const
{
    return (x*v.x + y*v.y + z*v.z + w*v.w);
}

template <typename type>
const vec4<type> vec4<type>::ZERO(0, 0, 0, 0);

template <typename type>
const vec4<type> vec4<type>::UNIT_X(1, 0, 0, 0);

template <typename type>
const vec4<type> vec4<type>::UNIT_Y(0, 1, 0, 0);

template <typename type>
const vec4<type> vec4<type>::UNIT_Z(0, 0, 1, 0);

template <typename type>
const vec4<type> vec4<type>::UNIT_W(0, 0, 0, 1);

#endif
