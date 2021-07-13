/*
 * @file transform.cpp 
 * @brief transform class, brief class representing a rigid euclidean transform as a quaternion and a vector
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

#include "transform_3d.h"

namespace PhysIKA {

template <typename Scalar>
Transform<Scalar, 3>::Transform()
    : translation_(Vector<Scalar, 3>(0, 0, 0)), rotation_(0, 0, 0, 1), scale_(Vector<Scalar, 3>(1, 1, 1))
{
}

template <typename Scalar>
Transform<Scalar, 3>::~Transform()
{
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const Vector<Scalar, 3>& translation)
    : translation_(translation), rotation_(0, 0, 0, 1), scale_(Vector<Scalar, 3>(1, 1, 1))
{
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const Quaternion<Scalar>& rotation)
    : translation_(0, 0, 0), rotation_(rotation), scale_(Vector<Scalar, 3>(1, 1, 1))
{
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const Vector<Scalar, 3>& translation, const Quaternion<Scalar>& rotation)
    : translation_(translation), rotation_(rotation), scale_(Vector<Scalar, 3>(1, 1, 1))
{
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const Vector<Scalar, 3>& translation, const Quaternion<Scalar>& rotation, const Vector<Scalar, 3>& scale)
    : translation_(translation), rotation_(rotation), scale_(scale)
{
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const SquareMatrix<Scalar, 4>& matrix)
{
    this->rotation_       = Quaternion<Scalar>(matrix);
    this->translation_[0] = matrix(0, 3);
    this->translation_[1] = matrix(1, 3);
    this->translation_[2] = matrix(2, 3);
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const SquareMatrix<Scalar, 3>& matrix)
{
    this->rotation_    = Quaternion<Scalar>(matrix);
    this->translation_ = Vector<Scalar, 3>(0, 0, 0);
    this->scale_       = Vector<Scalar, 3>(1, 1, 1);
}

template <typename Scalar>
Transform<Scalar, 3>::Transform(const Transform<Scalar, 3>& trans)
{
    this->translation_ = trans.translation();
    this->rotation_    = trans.rotation();
    this->scale_       = trans.scale();
}

template <typename Scalar>
Transform<Scalar, 3>& Transform<Scalar, 3>::operator=(const Transform<Scalar, 3>& trans)
{
    this->translation_ = trans.translation();
    this->rotation_    = trans.rotation();
    this->scale_       = trans.scale();
    return *this;
}

template <typename Scalar>
bool Transform<Scalar, 3>::operator==(const Transform<Scalar, 3>& trans) const
{
    return this->rotation_ == trans.rotation() && this->translation_ == trans.translation() && this->scale_ == trans.scale();
}

template <typename Scalar>
bool Transform<Scalar, 3>::operator!=(const Transform<Scalar, 3>& trans) const
{
    return !(this->rotation_ == trans.rotation() && this->translation_ == trans.translation() && this->scale_ == trans.scale());
}

template <typename Scalar>
SquareMatrix<Scalar, 3> Transform<Scalar, 3>::rotation3x3Matrix() const
{
    return this->rotation_.get3x3Matrix();
}

template <typename Scalar>
SquareMatrix<Scalar, 4> Transform<Scalar, 3>::rotation4x4Matrix() const
{
    return this->rotation_.get4x4Matrix();
}

template <typename Scalar>
SquareMatrix<Scalar, 4> Transform<Scalar, 3>::translation4x4Matrix() const
{
    return SquareMatrix<Scalar, 4>(1, 0, 0, this->translation_[0], 0, 1, 0, this->translation_[1], 0, 0, 1, this->translation_[2], 0, 0, 0, 1);
}
template <typename Scalar>
SquareMatrix<Scalar, 4> Transform<Scalar, 3>::scale4x4Matrix() const
{
    return SquareMatrix<Scalar, 4>(this->scale_[0], 0, 0, 0, 0, this->scale_[1], 0, 0, 0, 0, this->scale_[2], 0, 0, 0, 0, 1);
}

template <typename Scalar>
SquareMatrix<Scalar, 4> Transform<Scalar, 3>::transformMatrix() const
{
    SquareMatrix<Scalar, 4> matrix = this->rotation_.get4x4Matrix();
    matrix(0, 3)                   = this->translation_[0];
    matrix(1, 3)                   = this->translation_[1];
    matrix(2, 3)                   = this->translation_[2];
    SquareMatrix<Scalar, 4> scale_matrix(this->scale_[0], 0, 0, 0, 0, this->scale_[1], 0, 0, 0, 0, this->scale_[2], 0, 0, 0, 0, 1);

    return matrix * scale_matrix;
}

template <typename Scalar>
Vector<Scalar, 3> Transform<Scalar, 3>::rotate(const Vector<Scalar, 3>& input) const
{
    return this->rotation_.rotate(input);
}

template <typename Scalar>
Vector<Scalar, 3> Transform<Scalar, 3>::translate(const Vector<Scalar, 3>& input) const
{
    return this->translation_ + input;
}

template <typename Scalar>
Vector<Scalar, 3> Transform<Scalar, 3>::scaling(const Vector<Scalar, 3>& input) const
{
    Vector<Scalar, 3> tmp = input;
    tmp[0] *= this->scale_[0];
    tmp[1] *= this->scale_[1];
    tmp[2] *= this->scale_[2];
    return tmp;
}

template <typename Scalar>
Vector<Scalar, 3> Transform<Scalar, 3>::transform(const Vector<Scalar, 3>& input) const
{
    Vector<Scalar, 3> tmp = input;
    tmp[0] *= this->scale_[0];
    tmp[1] *= this->scale_[1];
    tmp[2] *= this->scale_[2];
    tmp = this->rotation_.rotate(tmp) + this->translation_;
    return tmp;
}

//explicit instantiation
template class Transform<float, 3>;
template class Transform<double, 3>;

}  //end of namespace PhysIKA
