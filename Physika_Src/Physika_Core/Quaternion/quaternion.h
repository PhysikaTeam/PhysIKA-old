/*
 * @file quaternion.h 
 * @brief quaternion class, it is used for rotation operations.
 * @author Sheng Yang
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
#include "Physika_Core/Vectors/vector_3d.h"

using Physika::Vector3D;

namespace Physika{

template <typename Scalar>
class Quaternion
{
public:
    Scalar x,y,z,w;
    
    /* Constructors */
    Quaternion();
    Quaternion(Scalar , Scalar , Scalar , Scalar );
    Quaternion(const Vector3D<Scalar> &, float );
    Quaternion(float, const Vector3D<Scalar> &);
    Quaternion(const Scalar *); 
    Quaternion(const Quaternion<Scalar> &);
    
    /* Assignment operators */
    Quaternion<Scalar> &operator = (const Quaternion<Scalar> &);
    Quaternion<Scalar> &operator += (const Quaternion<Scalar> &);
    Quaternion<Scalar> &operator -= (const Quaternion<Scalar> &);
    
    /* Special functions */
    Scalar norm();
    Quaternion<Scalar>& normalize();
    void set(const Vector3D<Scalar>&, Scalar );
    void set(Scalar , const Vector3D<Scalar>& );
  

    /* Operator overloading */
    Quaternion<Scalar> operator - (const Quaternion<Scalar> &);
    Quaternion<Scalar> operator - (void);
    Quaternion<Scalar> operator + (const Quaternion<Scalar> &);
    Quaternion<Scalar> operator * (Scalar );
    Quaternion<Scalar> operator / (Scalar );
    bool operator == (const Quaternion<Scalar> &);
    bool operator != (const Quaternion<Scalar> &);
    Scalar& operator[] (int);
    const Scalar& operator[] (int) const;

};


template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Quaternion<Scalar> &quat)
{
    s <<quat.x<<", "<<quat.y<<", "<<quat.z<<", "<<quat.w<<std::endl;
    return s; 
}



//make * operator commutative
template <typename S, typename T>
Quaternion<T> operator *(S scale, Quaternion<T> quad)
{
    return quad * scale;
}

namespace Type{
    typedef Quaternion<float> Quaternionf;
    typedef Quaternion<double> Quaterniond;
}

}//end of namespace Physika

#endif //PHSYIKA_CORE_QUATERNION_QUATERNION_H_