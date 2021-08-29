/*
 * @file quaternion.h 
 * @brief quaternion class, it is used for rotation operations.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHSYIKA_CORE_QUATERNION_QUATERNION_H_
#define PHSYIKA_CORE_QUATERNION_QUATERNION_H_

#include "../Vector.h"
#include "../Matrix.h"

namespace PhysIKA {

/*
 * Quaternion is defined for float and double.
 */

template <typename Real>
class Quaternion
{
public:
    /* Constructors */
    COMM_FUNC Quaternion();
    COMM_FUNC Quaternion(Real x, Real y, Real z, Real w);
    COMM_FUNC Quaternion(const Vector<Real, 3>& unit_axis, Real angle_rad);  //init from the rotation axis and angle(in radian)
    COMM_FUNC Quaternion(Real angle_rad, const Vector<Real, 3>& unit_axis);
    COMM_FUNC explicit Quaternion(const Real*);
    COMM_FUNC Quaternion(const Quaternion<Real>&);
    COMM_FUNC explicit Quaternion(const SquareMatrix<Real, 3>&);  //init from a 3x3matrix
    COMM_FUNC explicit Quaternion(const SquareMatrix<Real, 4>&);  //init from a 4x4matrix
    COMM_FUNC explicit Quaternion(const Vector<Real, 3>&);        //init form roll pitch yaw/ Euler angle;

    /* Assignment operators */
    COMM_FUNC Quaternion<Real>& operator=(const Quaternion<Real>&);
    COMM_FUNC Quaternion<Real>& operator+=(const Quaternion<Real>&);
    COMM_FUNC Quaternion<Real>& operator-=(const Quaternion<Real>&);

    /* Get and Set functions */
    COMM_FUNC inline Real x() const
    {
        return x_;
    }
    COMM_FUNC inline Real y() const
    {
        return y_;
    }
    COMM_FUNC inline Real z() const
    {
        return z_;
    }
    COMM_FUNC inline Real w() const
    {
        return w_;
    }

    COMM_FUNC inline void setX(const Real& x)
    {
        x_ = x;
    }
    COMM_FUNC inline void setY(const Real& y)
    {
        y_ = y;
    }
    COMM_FUNC inline void setZ(const Real& z)
    {
        z_ = z;
    }
    COMM_FUNC inline void setW(const Real& w)
    {
        w_ = w;
    }

    //rotate
    COMM_FUNC const Vector<Real, 3> rotate(const Vector<Real, 3>) const;  // rotates passed vec by this.
    COMM_FUNC void                  rotateVector(Vector<Real, 3>& v);
    COMM_FUNC void                  toRotationAxis(Real& rot, Vector<Real, 3>& axis) const;

    /* Special functions */
    COMM_FUNC Real norm();
    COMM_FUNC Quaternion<Real>& normalize();

    COMM_FUNC void set(const Vector<Real, 3>&, Real);
    COMM_FUNC void set(Real, const Vector<Real, 3>&);
    COMM_FUNC void set(const Vector<Real, 3>&);  //set from a euler angle.

    COMM_FUNC Real getAngle() const;                         // return the angle between this quat and the identity quaternion.
    COMM_FUNC Real getAngle(const Quaternion<Real>&) const;  // return the angle between this and the argument
    COMM_FUNC Quaternion<Real> getConjugate() const;         // return the conjugate

    COMM_FUNC SquareMatrix<Real, 3> get3x3Matrix() const;  //return 3x3matrix format
    COMM_FUNC SquareMatrix<Real, 4> get4x4Matrix() const;  //return 4x4matrix with a identity transform.
    COMM_FUNC Vector<Real, 3> getEulerAngle() const;

    COMM_FUNC Quaternion<Real> multiply_q(const Quaternion<Real>&);

    /* Operator overloading */
    COMM_FUNC Quaternion<Real> operator-(const Quaternion<Real>&) const;
    COMM_FUNC Quaternion<Real> operator-(void) const;
    COMM_FUNC Quaternion<Real> operator+(const Quaternion<Real>&) const;
    COMM_FUNC Quaternion<Real> operator*(const Quaternion<Real>&) const;
    COMM_FUNC Quaternion<Real> operator*(const Real&) const;
    COMM_FUNC Quaternion<Real> operator/(const Real&) const;
    COMM_FUNC bool             operator==(const Quaternion<Real>&) const;
    COMM_FUNC bool             operator!=(const Quaternion<Real>&) const;
    COMM_FUNC Real& operator[](unsigned int);
    COMM_FUNC const Real& operator[](unsigned int) const;
    COMM_FUNC Real        dot(const Quaternion<Real>&) const;

    COMM_FUNC static inline Quaternion<Real> Identity()
    {
        return Quaternion<Real>(0, 0, 0, 1);
    }

public:
    Real x_, y_, z_, w_;
};

//make * operator commutative
template <typename S, typename T>
COMM_FUNC inline Quaternion<T> operator*(S scale, const Quaternion<T>& quad)
{
    return quad * scale;
}

template class Quaternion<float>;
template class Quaternion<double>;
//convenient typedefs
typedef Quaternion<float>  Quaternionf;
typedef Quaternion<double> Quaterniond;

}  //end of namespace PhysIKA
#include "quaternion.inl"
#endif  //PHSYIKA_CORE_QUATERNION_QUATERNION_H_
