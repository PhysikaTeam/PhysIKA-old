/*
 * @file transform.cpp 
 * @brief transform class, brief class representing a rigid euclidean transform as a quaternion and a vector
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

#include "Physika_Core/Transform/transform.h"

namespace Physika{


template <typename Scalar>
Transform<Scalar>::Transform():translation_(Vector<Scalar, 3>(0,0,0)),
    rotation_(0,0,0,1),
    scale_(Vector<Scalar, 3>(1,1,1))
{

}

template <typename Scalar>
Transform<Scalar>::Transform(const Vector<Scalar,3> translation):
        translation_(translation),
        rotation_(0,0,0,1),
        scale_(Vector<Scalar, 3>(1,1,1))
{

}

template <typename Scalar>
Transform<Scalar>::Transform(const SquareMatrix<Scalar,4>& matrix)
{
    rotation_ = Quaternion<Scalar>(matrix);
    translation_[0] = matrix(0,3);
    translation_[1] = matrix(1,3);
    translation_[2] = matrix(2,3);
}

template <typename Scalar>
Transform<Scalar>::Transform(const SquareMatrix<Scalar, 3>& matrix)
{
    rotation_ = Quaternion<Scalar>(matrix);
    translation_ = Vector<Scalar, 3>(0,0,0);
    scale_ = Vector<Scalar, 3>(1,1,1);
}

template <typename Scalar>
Transform<Scalar>::Transform(const Quaternion<Scalar> rotation):
        rotation_(rotation),
        translation_(0,0,0),
        scale_(Vector<Scalar, 3>(1,1,1))
{

}

template <typename Scalar>
Transform<Scalar>::Transform(const Quaternion<Scalar>& rotation, const Vector<Scalar,3>& translation):
        translation_(translation),
        rotation_(rotation),
        scale_(Vector<Scalar, 3>(1,1,1))
{
    
}

template <typename Scalar>
Transform<Scalar>::Transform(const Quaternion<Scalar>& rotation, const Vector<Scalar,3>& translation, const Vector<Scalar, 3>& scale):
        translation_(translation),
        rotation_(rotation),
        scale_(scale)
{
    
}
template <typename Scalar>
Transform<Scalar>::Transform(const Vector<Scalar,3>& translation, const Quaternion<Scalar>& rotation):
        translation_(translation),
        rotation_(rotation),
       scale_(Vector<Scalar, 3>(1,1,1))
{


}

template <typename Scalar>
Transform<Scalar>::Transform(const Vector<Scalar,3>& translation, const Quaternion<Scalar>& rotation, const Vector<Scalar, 3>& scale):
        translation_(translation),
        rotation_(rotation),
        scale_(scale)
{
    
}

template <typename Scalar>
Vector<Scalar,3> Transform<Scalar>::transform(const Vector<Scalar, 3>& input) const
{
    Vector<Scalar, 3> tmp = this->rotation_.rotate(input) + this->translation_;
    tmp[0] *= scale_[0];
    tmp[1] *= scale_[1];
    tmp[2] *= scale_[2];
    return tmp;
}

//explicit instantiation
template class Transform<float>;
template class Transform<double>;

}  //end of namespace Physika
