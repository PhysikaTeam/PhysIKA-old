/*
 * @file quaternion.h 
 * @brief quaternion class, it is used for rotation operations.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHSYIKA_CORE_QUATERNION_QUATERNION_H_
#define PHSYIKA_CORE_QUATERNION_QUATERNION_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"


namespace Physika{

/*
 * Quaternion is defined for float and double.
 */

template <typename Scalar>
class Quaternion
{
public:    
    /* Constructors */
    Quaternion();
    Quaternion(Scalar x, Scalar y, Scalar z, Scalar w);
    Quaternion(const Vector<Scalar,3> &unit_axis, Scalar angle_rad);  //init from the rotation axis and angle(in radian)
    Quaternion(Scalar angle_rad, const Vector<Scalar,3> &unit_axis);
    Quaternion(const Scalar *); 
    Quaternion(const Quaternion<Scalar> &);
    Quaternion(const SquareMatrix<Scalar, 3> &);   //init from a 3x3matrix
    Quaternion(const SquareMatrix<Scalar,4> &);         //init from a 4x4matrix
    Quaternion(const Vector<Scalar, 3>& );         //init form roll pitch yaw/ Euler angle;
    
    /* Assignment operators */
    Quaternion<Scalar> &operator = (const Quaternion<Scalar> &);
    Quaternion<Scalar> &operator += (const Quaternion<Scalar> &);
    Quaternion<Scalar> &operator -= (const Quaternion<Scalar> &);
    
    /* Get and Set functions */
    inline Scalar x() const { return x_;}
    inline Scalar y() const { return y_;}
    inline Scalar z() const { return z_;}
    inline Scalar w() const { return w_;}

    inline void setX(const Scalar& x) { x_ = x;}
    inline void setY(const Scalar& y) { y_ = y;}
    inline void setZ(const Scalar& z) { z_ = z;}
    inline void setW(const Scalar& w) { w_ = w;}

    /* Special functions */
    Scalar norm();
    Quaternion<Scalar>& normalize();
    void set(const Vector<Scalar,3>&, Scalar );
    void set(Scalar , const Vector<Scalar,3>& );
    void set(const Vector<Scalar,3>& );                              //set from a euler angle.
    Scalar getAngle() const;                                         // return the angle between this quat and the identity quaternion.
    Scalar getAngle(const Quaternion<Scalar>&) const;                // return the angle between this and the argument
    Scalar dot(const Quaternion<Scalar> &) const;
    Quaternion<Scalar> getConjugate() const;                         // return the conjugate
    const Vector<Scalar,3> rotate(const Vector<Scalar,3> ) const;    // rotates passed vec by this;
    SquareMatrix<Scalar, 3> get3x3Matrix() const;  //return 3x3matrix format
    SquareMatrix<Scalar, 4> get4x4Matrix() const;        //return 4x4matrix with a identity transform.
    Vector<Scalar, 3> getEulerAngle() const;
    void toRadiansAndUnitAxis(Scalar& angle, Vector<Scalar, 3>& axis) const;
  

    /* Operator overloading */
    Quaternion<Scalar> operator - (const Quaternion<Scalar>& );
    Quaternion<Scalar> operator - (void);
    Quaternion<Scalar> operator + (const Quaternion<Scalar>& );
    Quaternion<Scalar> operator * (const Quaternion<Scalar>& ) const;
    Quaternion<Scalar> operator * (const Scalar& );
    Quaternion<Scalar> operator / (const Scalar& );
    bool operator == (const Quaternion<Scalar>& );
    bool operator != (const Quaternion<Scalar>& );
    Scalar& operator[] (unsigned int);
    const Scalar& operator[] (unsigned int) const;

    static inline Quaternion<Scalar> identityQuaternion() { return Quaternion<Scalar>(0,0,0,1); }

protected:
    Scalar x_,y_,z_,w_;
protected:
    PHYSIKA_STATIC_ASSERT((is_same<Scalar,float>::value||is_same<Scalar,double>::value),
                           "Quaternion<Scalar> are only defined for Scalar type of float and double");
};


template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const Quaternion<Scalar> &quat)
{
    s <<quat.x()<<", "<<quat.y()<<", "<<quat.z()<<", "<<quat.w()<<std::endl;
    return s; 
}

//make * operator commutative
template <typename S, typename T>
inline Quaternion<T> operator *(S scale, const Quaternion<T> &quad)
{
    return quad * scale;
}

//convenient typedefs
typedef Quaternion<float> Quaternionf;
typedef Quaternion<double> Quaterniond;

}//end of namespace Physika

#endif //PHSYIKA_CORE_QUATERNION_QUATERNION_H_
