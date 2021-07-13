/*
 * @file transform_2d.h 
 * @brief transform class, brief class representing a rigid 2d transform use matrix and vector
 * @author Sheng Yang
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include "transform_2d.h"

namespace PhysIKA {

template <typename Scalar>
Transform<Scalar, 2>::~Transform()
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform()
    : translation_(Vector<Scalar, 2>(0)), rotate_angle_(0), scale_(Vector<Scalar, 2>(1))
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Vector<Scalar, 2>& translation)
    : translation_(translation), rotate_angle_(0), scale_(Vector<Scalar, 2>(1))
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(Scalar rotate_angle)
    : translation_(Vector<Scalar, 2>(0)), rotate_angle_(rotate_angle), scale_(Vector<Scalar, 2>(1))
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const SquareMatrix<Scalar, 2>& rotation)
    : translation_(Vector<Scalar, 2>(0)), scale_(Vector<Scalar, 2>(1))
{
    this->rotate_angle_ = acos(rotation(0, 0));
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Vector<Scalar, 2>& translation, const SquareMatrix<Scalar, 2>& rotation)
    : translation_(translation), scale_(Vector<Scalar, 2>(1))
{
    this->rotate_angle_ = acos(rotation(0, 0));
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Vector<Scalar, 2>& translation, Scalar rotate_angle)
    : translation_(translation), rotate_angle_(rotate_angle), scale_(Vector<Scalar, 2>(1))
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Vector<Scalar, 2>& translation, const SquareMatrix<Scalar, 2>& rotation, const Vector<Scalar, 2>& scale)
    : translation_(translation), scale_(scale)
{
    this->rotate_angle_ = acos(rotation(0, 0));
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Vector<Scalar, 2>& translation, Scalar rotate_angle, const Vector<Scalar, 2>& scale)
    : translation_(translation), rotate_angle_(rotate_angle), scale_(Vector<Scalar, 2>(1))
{
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const SquareMatrix<Scalar, 3>& matrix)
{
    this->rotate_angle_   = acos(matrix(0, 0));
    this->translation_[0] = matrix(0, 2);
    this->translation_[1] = matrix(1, 2);
}

template <typename Scalar>
Transform<Scalar, 2>::Transform(const Transform<Scalar, 2>& trans)
{
    this->rotate_angle_ = trans.rotateAngle();
    this->translation_  = trans.translation();
    this->scale_        = trans.scale();
}

template <typename Scalar>
Transform<Scalar, 2>& Transform<Scalar, 2>::operator=(const Transform<Scalar, 2>& trans)
{
    this->rotate_angle_ = trans.rotateAngle();
    this->translation_  = trans.translation();
    this->scale_        = trans.scale();
    return *this;
}

template <typename Scalar>
bool Transform<Scalar, 2>::operator==(const Transform<Scalar, 2>& trans) const
{
    return this->rotate_angle_ == trans.rotateAngle() && this->translation_ == trans.translation() && this->scale_ == trans.scale();
}

template <typename Scalar>
bool Transform<Scalar, 2>::operator!=(const Transform<Scalar, 2>& trans) const
{
    return !(this->rotate_angle_ == trans.rotateAngle() && this->translation_ == trans.translation() && this->scale_ == trans.scale());
}

template <typename Scalar>
SquareMatrix<Scalar, 2> Transform<Scalar, 2>::rotation2x2Matrix() const
{
    return SquareMatrix<Scalar, 2>(cos(this->rotate_angle_), -sin(this->rotate_angle_), sin(this->rotate_angle_), cos(this->rotate_angle_));
}

template <typename Scalar>
SquareMatrix<Scalar, 3> Transform<Scalar, 2>::rotation3x3Matrix() const
{
    return SquareMatrix<Scalar, 3>(cos(this->rotate_angle_), -sin(this->rotate_angle_), 0, sin(this->rotate_angle_), cos(this->rotate_angle_), 0, 0, 0, 1);
}

template <typename Scalar>
SquareMatrix<Scalar, 3> Transform<Scalar, 2>::scale3x3Matrix() const
{
    return SquareMatrix<Scalar, 3>(this->scale_[0], 0, 0, 0, this->scale_[1], 0, 0, 0, 1);
}

template <typename Scalar>
SquareMatrix<Scalar, 3> Transform<Scalar, 2>::transformMatrix() const
{
    SquareMatrix<Scalar, 3> matrix       = this->rotation3x3Matrix();
    matrix(0, 1)                         = this->translation_[0];
    matrix(0, 2)                         = this->translation_[1];
    SquareMatrix<Scalar, 3> scale_matrix = this->scale3x3Matrix();
    return matrix * scale_matrix;
}

template <typename Scalar>
SquareMatrix<Scalar, 3> Transform<Scalar, 2>::translation3x3Matrix() const
{
    return SquareMatrix<Scalar, 3>(1, 0, this->translation_[0], 0, 1, this->translation_[1], 0, 0, 1);
}

template <typename Scalar>
Vector<Scalar, 2> Transform<Scalar, 2>::rotate(const Vector<Scalar, 2>& input) const
{
    Scalar x, y;
    x = input[0] * cos(this->rotate_angle_) - input[1] * sin(this->rotate_angle_);
    y = input[0] * sin(this->rotate_angle_) + input[1] * cos(this->rotate_angle_);
    return Vector<Scalar, 2>(x, y);
}

template <typename Scalar>
Vector<Scalar, 2> Transform<Scalar, 2>::translate(const Vector<Scalar, 2>& input) const
{
    return this->translation_ + input;
}

template <typename Scalar>
Vector<Scalar, 2> Transform<Scalar, 2>::scaling(const Vector<Scalar, 2>& input) const
{
    Vector<Scalar, 2> tmp = input;
    tmp[0] *= this->scale_[0];
    tmp[1] *= this->scale_[1];
    return tmp;
}

template <typename Scalar>
Vector<Scalar, 2> Transform<Scalar, 2>::transform(const Vector<Scalar, 2>& input) const
{
    Vector<Scalar, 2> tmp = input;
    tmp[0] *= this->scale_[0];
    tmp[1] *= this->scale_[1];
    tmp = this->rotate(tmp) + this->translation_;
    return tmp;
}

//explicit instantiation
template class Transform<float, 2>;
template class Transform<double, 2>;

}  //end of namespace PhysIKA
