#ifndef _MAT4_H_
#define _MAT4_H_

template <typename type> class mat4
{
protected:
    union {
        type m[4][4];
        type _m[16];
    };

public:
    mat4();

    mat4(type m00, type m01, type m02, type m03,
         type m10, type m11, type m12, type m13,
         type m20, type m21, type m22, type m23,
         type m30, type m31, type m32, type m33);

    const type* coefficients() const;

    const type* operator[](int iRow) const;

    bool operator==(const mat4& m2) const;

    bool operator!=(const mat4& m2) const;

    mat4 operator+(const mat4& m2) const;

    mat4 operator-(const mat4& m2) const;

    mat4 operator*(const mat4& m2) const;

    vec4<type> operator*(const vec4<type>& v) const;

    mat4 operator*(type f) const;

    mat4 transpose(void) const;

    mat4 adjoint() const;

    mat4 inverse() const;

    float determinant() const;

    static mat4 rotatex(type angle);

    static mat4 rotatey(type angle);

    static mat4 rotatez(type angle);

    static mat4 perspectiveProjection(type fovy, type aspect, type zNear, type zFar);

    static mat4 orthoProjection(type xRight, type xLeft, type yTop, type yBottom, type zNear, type zFar);

    template <class t>
    mat4<t> cast() {
        return mat4<t>((t) m[0][0], (t) m[0][1], (t) m[0][2], (t) m[0][3],
            (t) m[1][0], (t)m[1][1], (t) m[1][2], (t) m[1][3],
            (t) m[2][0], (t)m[2][1], (t) m[2][2], (t) m[2][3],
            (t) m[3][0], (t)m[3][1], (t) m[3][2], (t) m[3][3]
        );
    }

    static const mat4 ZERO;

    static const mat4 IDENTITY;
};

typedef mat4<float> mat4f;

typedef mat4<double> mat4d;

template <typename type>
inline mat4<type>::mat4()
{
}

template <typename type>
inline mat4<type>::mat4(type m00, type m01, type m02, type m03,
                        type m10, type m11, type m12, type m13,
                        type m20, type m21, type m22, type m23,
                        type m30, type m31, type m32, type m33)
{
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
}

template <typename type>
inline const type* mat4<type>::coefficients() const
{
    return _m;
}

template <typename type>
inline const type* mat4<type>::operator[](int iRow) const
{
    return m[iRow];
}

template <typename type>
inline bool mat4<type>::operator==(const mat4<type>& m2) const
{
    if (
        m[0][0] != m2.m[0][0] || m[0][1] != m2.m[0][1] || m[0][2] != m2.m[0][2] || m[0][3] != m2.m[0][3] ||
        m[1][0] != m2.m[1][0] || m[1][1] != m2.m[1][1] || m[1][2] != m2.m[1][2] || m[1][3] != m2.m[1][3] ||
        m[2][0] != m2.m[2][0] || m[2][1] != m2.m[2][1] || m[2][2] != m2.m[2][2] || m[2][3] != m2.m[2][3] ||
        m[3][0] != m2.m[3][0] || m[3][1] != m2.m[3][1] || m[3][2] != m2.m[3][2] || m[3][3] != m2.m[3][3])
        return false;

    return true;
}

template <typename type>
inline bool mat4<type>::operator!=(const mat4<type>& m2) const
{
    if (
        m[0][0] != m2.m[0][0] || m[0][1] != m2.m[0][1] || m[0][2] != m2.m[0][2] || m[0][3] != m2.m[0][3] ||
        m[1][0] != m2.m[1][0] || m[1][1] != m2.m[1][1] || m[1][2] != m2.m[1][2] || m[1][3] != m2.m[1][3] ||
        m[2][0] != m2.m[2][0] || m[2][1] != m2.m[2][1] || m[2][2] != m2.m[2][2] || m[2][3] != m2.m[2][3] ||
        m[3][0] != m2.m[3][0] || m[3][1] != m2.m[3][1] || m[3][2] != m2.m[3][2] || m[3][3] != m2.m[3][3])
        return true;

    return false;
}

template <typename type>
inline mat4<type> mat4<type>::operator+(const mat4<type>& m2) const
{
    mat4<type> r;

    r.m[0][0] = m[0][0] + m2.m[0][0];
    r.m[0][1] = m[0][1] + m2.m[0][1];
    r.m[0][2] = m[0][2] + m2.m[0][2];
    r.m[0][3] = m[0][3] + m2.m[0][3];

    r.m[1][0] = m[1][0] + m2.m[1][0];
    r.m[1][1] = m[1][1] + m2.m[1][1];
    r.m[1][2] = m[1][2] + m2.m[1][2];
    r.m[1][3] = m[1][3] + m2.m[1][3];

    r.m[2][0] = m[2][0] + m2.m[2][0];
    r.m[2][1] = m[2][1] + m2.m[2][1];
    r.m[2][2] = m[2][2] + m2.m[2][2];
    r.m[2][3] = m[2][3] + m2.m[2][3];

    r.m[3][0] = m[3][0] + m2.m[3][0];
    r.m[3][1] = m[3][1] + m2.m[3][1];
    r.m[3][2] = m[3][2] + m2.m[3][2];
    r.m[3][3] = m[3][3] + m2.m[3][3];

    return r;
}

template <typename type>
inline mat4<type> mat4<type>::operator-(const mat4<type>& m2) const
{
    mat4 r;
    r.m[0][0] = m[0][0] - m2.m[0][0];
    r.m[0][1] = m[0][1] - m2.m[0][1];
    r.m[0][2] = m[0][2] - m2.m[0][2];
    r.m[0][3] = m[0][3] - m2.m[0][3];

    r.m[1][0] = m[1][0] - m2.m[1][0];
    r.m[1][1] = m[1][1] - m2.m[1][1];
    r.m[1][2] = m[1][2] - m2.m[1][2];
    r.m[1][3] = m[1][3] - m2.m[1][3];

    r.m[2][0] = m[2][0] - m2.m[2][0];
    r.m[2][1] = m[2][1] - m2.m[2][1];
    r.m[2][2] = m[2][2] - m2.m[2][2];
    r.m[2][3] = m[2][3] - m2.m[2][3];

    r.m[3][0] = m[3][0] - m2.m[3][0];
    r.m[3][1] = m[3][1] - m2.m[3][1];
    r.m[3][2] = m[3][2] - m2.m[3][2];
    r.m[3][3] = m[3][3] - m2.m[3][3];

    return r;
}

template <typename type>
inline mat4<type> mat4<type>::operator*(const mat4<type>& m2) const
{
    mat4 r;
    r.m[0][0] = m[0][0] * m2.m[0][0] + m[0][1] * m2.m[1][0] + m[0][2] * m2.m[2][0] + m[0][3] * m2.m[3][0];
    r.m[0][1] = m[0][0] * m2.m[0][1] + m[0][1] * m2.m[1][1] + m[0][2] * m2.m[2][1] + m[0][3] * m2.m[3][1];
    r.m[0][2] = m[0][0] * m2.m[0][2] + m[0][1] * m2.m[1][2] + m[0][2] * m2.m[2][2] + m[0][3] * m2.m[3][2];
    r.m[0][3] = m[0][0] * m2.m[0][3] + m[0][1] * m2.m[1][3] + m[0][2] * m2.m[2][3] + m[0][3] * m2.m[3][3];

    r.m[1][0] = m[1][0] * m2.m[0][0] + m[1][1] * m2.m[1][0] + m[1][2] * m2.m[2][0] + m[1][3] * m2.m[3][0];
    r.m[1][1] = m[1][0] * m2.m[0][1] + m[1][1] * m2.m[1][1] + m[1][2] * m2.m[2][1] + m[1][3] * m2.m[3][1];
    r.m[1][2] = m[1][0] * m2.m[0][2] + m[1][1] * m2.m[1][2] + m[1][2] * m2.m[2][2] + m[1][3] * m2.m[3][2];
    r.m[1][3] = m[1][0] * m2.m[0][3] + m[1][1] * m2.m[1][3] + m[1][2] * m2.m[2][3] + m[1][3] * m2.m[3][3];

    r.m[2][0] = m[2][0] * m2.m[0][0] + m[2][1] * m2.m[1][0] + m[2][2] * m2.m[2][0] + m[2][3] * m2.m[3][0];
    r.m[2][1] = m[2][0] * m2.m[0][1] + m[2][1] * m2.m[1][1] + m[2][2] * m2.m[2][1] + m[2][3] * m2.m[3][1];
    r.m[2][2] = m[2][0] * m2.m[0][2] + m[2][1] * m2.m[1][2] + m[2][2] * m2.m[2][2] + m[2][3] * m2.m[3][2];
    r.m[2][3] = m[2][0] * m2.m[0][3] + m[2][1] * m2.m[1][3] + m[2][2] * m2.m[2][3] + m[2][3] * m2.m[3][3];

    r.m[3][0] = m[3][0] * m2.m[0][0] + m[3][1] * m2.m[1][0] + m[3][2] * m2.m[2][0] + m[3][3] * m2.m[3][0];
    r.m[3][1] = m[3][0] * m2.m[0][1] + m[3][1] * m2.m[1][1] + m[3][2] * m2.m[2][1] + m[3][3] * m2.m[3][1];
    r.m[3][2] = m[3][0] * m2.m[0][2] + m[3][1] * m2.m[1][2] + m[3][2] * m2.m[2][2] + m[3][3] * m2.m[3][2];
    r.m[3][3] = m[3][0] * m2.m[0][3] + m[3][1] * m2.m[1][3] + m[3][2] * m2.m[2][3] + m[3][3] * m2.m[3][3];

    return r;
}

template <typename type>
inline vec4<type> mat4<type>::operator*(const vec4<type>& v) const
{
    return vec4<type>(
               m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
               m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
               m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
               m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
           );
}

template <typename type>
inline mat4<type> mat4<type>::operator*(type f) const
{
    mat4<type> r;

    r.m[0][0] = m[0][0] * f;
    r.m[0][1] = m[0][1] * f;
    r.m[0][2] = m[0][2] * f;
    r.m[0][3] = m[0][3] * f;

    r.m[1][0] = m[1][0] * f;
    r.m[1][1] = m[1][1] * f;
    r.m[1][2] = m[1][2] * f;
    r.m[1][3] = m[1][3] * f;

    r.m[2][0] = m[2][0] * f;
    r.m[2][1] = m[2][1] * f;
    r.m[2][2] = m[2][2] * f;
    r.m[2][3] = m[2][3] * f;

    r.m[3][0] = m[3][0] * f;
    r.m[3][1] = m[3][1] * f;
    r.m[3][2] = m[3][2] * f;
    r.m[3][3] = m[3][3] * f;

    return r;
}

template <typename type>
inline mat4<type> mat4<type>::transpose(void) const
{
    return mat4(m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
                m[0][3], m[1][3], m[2][3], m[3][3]);
}

template <typename type>
inline static type
MINOR(const mat4<type>& m, int r0, int r1, int r2, int c0, int c1, int c2)
{
    return type(    m[r0][c0] *(m[r1][c1] * m[r2][c2] - m[r2][c1] * m[r1][c2]) -
                    m[r0][c1] *(m[r1][c0] * m[r2][c2] - m[r2][c0] * m[r1][c2]) +
                    m[r0][c2] *(m[r1][c0] * m[r2][c1] - m[r2][c0] * m[r1][c1]) );
}

template <typename type>
mat4<type> mat4<type>::adjoint() const
{
    return mat4(MINOR(*this, 1, 2, 3, 1, 2, 3),
                -MINOR(*this, 0, 2, 3, 1, 2, 3),
                MINOR(*this, 0, 1, 3, 1, 2, 3),
                -MINOR(*this, 0, 1, 2, 1, 2, 3),

                -MINOR(*this, 1, 2, 3, 0, 2, 3),
                MINOR(*this, 0, 2, 3, 0, 2, 3),
                -MINOR(*this, 0, 1, 3, 0, 2, 3),
                MINOR(*this, 0, 1, 2, 0, 2, 3),

                MINOR(*this, 1, 2, 3, 0, 1, 3),
                -MINOR(*this, 0, 2, 3, 0, 1, 3),
                MINOR(*this, 0, 1, 3, 0, 1, 3),
                -MINOR(*this, 0, 1, 2, 0, 1, 3),

                -MINOR(*this, 1, 2, 3, 0, 1, 2),
                MINOR(*this, 0, 2, 3, 0, 1, 2),
                -MINOR(*this, 0, 1, 3, 0, 1, 2),
                MINOR(*this, 0, 1, 2, 0, 1, 2));
}

template <typename type>
mat4<type> mat4<type>::inverse() const
{
    return adjoint() * (1.0f / determinant());
}

template <typename type>
float mat4<type>::determinant() const
{
    return float(    m[0][0] * MINOR(*this, 1, 2, 3, 1, 2, 3) -
                    m[0][1] * MINOR(*this, 1, 2, 3, 0, 2, 3) +
                    m[0][2] * MINOR(*this, 1, 2, 3, 0, 1, 3) -
                    m[0][3] * MINOR(*this, 1, 2, 3, 0, 1, 2) );
}

template <typename type>
inline mat4<type> mat4<type>::rotatex(type angle)
{
    type ca = (type) cos(angle * M_PI / 180.0);
    type sa = (type) sin(angle * M_PI / 180.0);
    return mat4<type>(1, 0, 0, 0,
                      0, ca, -sa, 0,
                      0, sa, ca, 0,
                      0, 0, 0, 1);
}

template <typename type>
inline mat4<type> mat4<type>::rotatey(type angle)
{
    type ca = (type) cos(angle * M_PI / 180.0);
    type sa = (type) sin(angle * M_PI / 180.0);
    return mat4<type>(ca, 0, sa, 0,
                      0, 1, 0, 0,
                      -sa, 0, ca, 0,
                      0, 0, 0, 1);
}

template <typename type>
inline mat4<type> mat4<type>::rotatez(type angle)
{
    type ca = (type) cos(angle * M_PI / 180.0);
    type sa = (type) sin(angle * M_PI / 180.0);
    return mat4<type>(ca, -sa, 0, 0,
                      sa, ca, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
}

template <typename type>
inline mat4<type> mat4<type>::perspectiveProjection(type fovy, type aspect, type zNear, type zFar)
{
    type f = (type) 1 / tan(fovy * M_PI / 180.0f / 2);
    return mat4<type>(f / aspect, 0, 0,                         0,
                      0,        f, 0,                         0,
                      0,        0, (zFar + zNear) / (zNear - zFar), (2*zFar*zNear) / (zNear - zFar),
                      0,        0, -1,                        0);
}

template <typename type>
inline mat4<type> mat4<type>::orthoProjection(type xRight, type xLeft, type yTop, type yBottom, type zNear, type zFar)
{
    type tx, ty, tz;
    tx = - (xRight + xLeft) / (xRight - xLeft);
    ty = - (yTop + yBottom) / (yTop - yBottom);
    tz = - (zFar + zNear) / (zFar - zNear);
    return mat4f::mat4(2 / (xRight - xLeft), 0,                  0,                tx,
                       0,                  2 / (yTop - yBottom), 0,                ty,
                       0,                  0,                 -2 / (zFar - zNear), tz,
                       0,                  0,                  0,                1);
}

template <typename type>
const mat4<type> mat4<type>::ZERO(
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0);

template <typename type>
const mat4<type> mat4<type>::IDENTITY(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1);

#endif
