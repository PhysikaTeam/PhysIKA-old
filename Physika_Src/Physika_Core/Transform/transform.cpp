/*
 * @file transform.cpp 
 * @brief transform class, brief class representing a rigid euclidean transform as a quaternion and a vector
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
#include "Physika_Core/Transform/transform.h"

namespace Physika{

template <typename Scalar>
Transform<Scalar>::Transform(const Vector<Scalar,3> position):
        position_(position),
        orientation_(0,0,0,1)
{

}

template <typename Scalar>
Transform<Scalar>::Transform(const Quaternion<Scalar> orientation):
        orientation_(orientation),
        position_(0,0,0)
{

}

template <typename Scalar>
Transform<Scalar>::Transform(const Quaternion<Scalar>& orientation, const Vector<Scalar,3>& position):
        position_(position),
        orientation_(orientation)
{
    
}

template <typename Scalar>
Transform<Scalar>::Transform(const Vector<Scalar,3>& position, const Quaternion<Scalar>& orientation):
        position_(position),
        orientation_(orientation)
{

}

//explicit instantiation
template class Transform<float>;
template class Transform<double>;

}  //end of namespace Physika
