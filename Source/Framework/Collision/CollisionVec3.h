/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: device bvh structure
 * @version    : 1.0
 */

#pragma once

#include <iostream>

namespace PhysIKA {
/**
     * vec3f data structure
     */
class vec3f
{
public:
    union
    {
        struct
        {
            float x, y, z;  //!< x, y, z of 3d-vector
        };
        struct
        {
            float v[3];  //!< data of 3d-vector
        };
    };

    /**
         * constructor
         */
    vec3f()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    /**
         * copy constructor
         *
         * @param[in] v another vec3f
         */
    vec3f(const vec3f& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    /**
         * constructor
         *
         * @param[in] v data pointer
         */
    vec3f(const float* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    /**
         * constructor
         *
         * @param[in] v data pointer
         */
    vec3f(float* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    /**
         * constructor
         *
         * @param[in] x x-direction data
         * @param[in] y y-direction data
         * @param[in] z z-direction data
         */
    vec3f(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    /**
         * operator[]
         *
         * @param[in] i index
         * @return the i-th data of the vec3f
         */
    float operator[](int i) const
    {
        return v[i];
    }

    /**
         * operator[]
         *
         * @param[in] i index
         * @return the i-th data of the vec3f
         */
    float& operator[](int i)
    {
        return v[i];
    }

    /**
         * operator += add the vec3f with another one
         *
         * @param[in] v another vec3f
         * @return add result
         */
    vec3f& operator+=(const vec3f& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    /**
         * operator -= substract the vec3f with another one
         *
         * @param[in] v another vec3f
         * @return substract result
         */
    vec3f& operator-=(const vec3f& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    /**
         * operator *= multiply with a scalar
         *
         * @param[in] t scalar
         * @return multiply result
         */
    vec3f& operator*=(float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    /**
         * operator /= divide with a scalar
         *
         * @param[in] t scalar
         * @return divide result
         */
    vec3f& operator/=(float t)
    {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    /**
         * negate the current vector
         */
    void negate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    /**
         * negate the current vector
         *
         * @return the negated result
         */
    vec3f operator-() const
    {
        return vec3f(-x, -y, -z);
    }

    /**
         * vector add two vector
         *
         * @param[in] v another vec3f
         * @return add result
         */
    vec3f operator+(const vec3f& v) const
    {
        return vec3f(x + v.x, y + v.y, z + v.z);
    }

    /**
         * vector substract
         *
         * @param[in] v another vec3f
         * @return the substracted result
         */
    vec3f operator-(const vec3f& v) const
    {
        return vec3f(x - v.x, y - v.y, z - v.z);
    }

    /**
         * vector multiply a scalar
         *
         * @param[in] t scaler
         * @return the multiplied result
         */
    vec3f operator*(float t) const
    {
        return vec3f(x * t, y * t, z * t);
    }

    /**
         * vector divide a scalar
         *
         * @param[in] t scaler
         * @return the divided result
         */
    vec3f operator/(float t) const
    {
        return vec3f(x / t, y / t, z / t);
    }

    /**
         * vector cross product
         *
         * @param[in] v another vec3f
         * @return cross product result
         */
    const vec3f cross(const vec3f& vec) const
    {
        return vec3f(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
    }

    /**
         * vector dot product
         *
         * @param[in] v another vec3f
         * @return dot product result
         */
    float dot(const vec3f& vec) const
    {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    /**
         * vector normalize
         */
    void normalize()
    {
        float sum = x * x + y * y + z * z;
        if (sum > float(10e-12))
        {
            float base = float(1.0 / sqrt(sum));
            x *= base;
            y *= base;
            z *= base;
        }
    }

    /**
         * get vector length
         *
         * @return the length of the vector
         */
    float length() const
    {
        return float(sqrt(x * x + y * y + z * z));
    }

    /**
         * get unit vector of the current vector
         *
         * @return the unit vector
         */
    vec3f getUnit() const
    {
        return (*this) / length();
    }

    /**
         * eplision equal
         *
         * @param[in] a one number
         * @param[in] b the other number
         * @param[in] tol threshold
         * @return whether the two number is equal with respect to the current threshold
         */
    inline bool isEqual(float a, float b, float tol = float(10e-6)) const
    {
        return fabs(a - b) < tol;
    }

    /**
         * check unit
         *
         * @return whether the current vector is a unit vector
         */
    bool isUnit() const
    {
        return isEqual(squareLength(), 1.f);
    }

    /**
         * get the infinity norm
         *
         * @return the infinity norm
         */
    float infinityNorm() const
    {
        return fmax(fmax(fabs(x), fabs(y)), fabs(z));
    }

    /**
         * set the value of the vec3f
         *
         * @param[in] vx x-direction value
         * @param[in] vy y-direction value
         * @param[in] vz z-direction value
         * @return new vec3f
         */
    vec3f& set_value(const float& vx, const float& vy, const float& vz)
    {
        x = vx;
        y = vy;
        z = vz;
        return *this;
    }

    /**
         * check if two vec3f has the same value
         *
         * @param[in] other another vector
         * @return check result
         */
    bool equal_abs(const vec3f& other)
    {
        return x == other.x && y == other.y && z == other.z;
    }

    /**
         * get the square length
         *
         * @return the square length
         */
    float squareLength() const
    {
        return x * x + y * y + z * z;
    }

    /**
         * get a vec3f with all elements 0
         *
         * @return the zero vector
         */
    static vec3f zero()
    {
        return vec3f(0.f, 0.f, 0.f);
    }

    //! Named constructor: retrieve vector for nth axis
    static vec3f axis(int n)
    {
        switch (n)
        {
            case 0: {
                return xAxis();
            }
            case 1: {
                return yAxis();
            }
            case 2: {
                return zAxis();
            }
        }
        return vec3f();
    }

    //! Named constructor: retrieve vector for x axis
    static vec3f xAxis()
    {
        return vec3f(1.f, 0.f, 0.f);
    }
    //! Named constructor: retrieve vector for y axis
    static vec3f yAxis()
    {
        return vec3f(0.f, 1.f, 0.f);
    }
    //! Named constructor: retrieve vector for z axis
    static vec3f zAxis()
    {
        return vec3f(0.f, 0.f, 1.f);
    }
};

/**
     * scalar multiply vec3f
     *
     * @param[in] t scalar
     * @param[in] v vec3f
     * @return result
     */
inline vec3f operator*(float t, const vec3f& v)
{
    return vec3f(v.x * t, v.y * t, v.z * t);
}

/**
     * lerp two vec3f with (1 - t) * a + t * b
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @param[in] t scalar
     * @return lerp result
     */
inline vec3f interp(const vec3f& a, const vec3f& b, float t)
{
    return a * (1 - t) + b * t;
}

/**
     * vinterp two vec3f with t * a + (1 - t) * b
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @param[in] t scalar
     * @return vinterp result
     */
inline vec3f vinterp(const vec3f& a, const vec3f& b, float t)
{
    return a * t + b * (1 - t);
}

/**
     * calculate weighted position of three vec3f
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @param[in] c vec3f
     * @param[in] u weight
     * @param[in] v weight
     * @param[in] w weight
     * @return weighted position result
     */
inline vec3f interp(const vec3f& a, const vec3f& b, const vec3f& c, float u, float v, float w)
{
    return a * u + b * v + c * w;
}

/**
     * calculate the distance of to points
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @return distance
     */
inline float vdistance(const vec3f& a, const vec3f& b)
{
    return (a - b).length();
}

/**
     * print vec3f data
     * @param[in] os output stream
     * @param[in] v  vec3f
     * @return output stream
     */
inline std::ostream& operator<<(std::ostream& os, const vec3f& v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    return os;
}

/**
     * get the min value of two vec3f in all directions
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @return the min values in vec3f
     */
inline void vmin(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmin(a[0], b[0]),
        fmin(a[1], b[1]),
        fmin(a[2], b[2]));
}

/**
     * get the max value of two vec3f in all directions
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @return the max values in vec3f
     */
inline void vmax(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmax(a[0], b[0]),
        fmax(a[1], b[1]),
        fmax(a[2], b[2]));
}

/**
     * lerp between two vector
     * @param[in] a vec3f
     * @param[in] b vec3f
     * @param[in] t scalar
     * @return lerp result
     */
inline vec3f lerp(const vec3f& a, const vec3f& b, float t)
{
    return a + t * (b - a);
}
}  // namespace PhysIKA