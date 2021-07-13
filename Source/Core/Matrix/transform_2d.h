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

#ifndef PHSYIKA_CORE_TRANSFORM_TRANSFORM_2D_H_
#define PHSYIKA_CORE_TRANSFORM_TRANSFORM_2D_H_

#include "../Vector.h"
#include "../Matrix.h"
#include "../Utility.h"
#include "Core/Quaternion/quaternion.h"
#include "transform.h"

namespace PhysIKA {

/*
 * Transform is defined for float and double
 */

template <typename Scalar>
class Transform<Scalar, 2>
{
public:
    /* Constructions */

    Transform();
    ~Transform();
    explicit Transform(const Vector<Scalar, 2>& translation);
    explicit Transform(Scalar rotate_angle);
    explicit Transform(const SquareMatrix<Scalar, 2>& rotation);  //rotation is represented by a matrix;
    Transform(const Vector<Scalar, 2>& translation, const SquareMatrix<Scalar, 2>& rotation);
    Transform(const Vector<Scalar, 2>& translation, Scalar rotate_angle);
    Transform(const Vector<Scalar, 2>& translation, const SquareMatrix<Scalar, 2>& rotation, const Vector<Scalar, 2>& scale);
    Transform(const Vector<Scalar, 2>& translation, Scalar rotate_angle, const Vector<Scalar, 2>& scale);

    Transform(const SquareMatrix<Scalar, 3>& matrix);  //Init transform from a matrix.now it's not all right.Suggest not use this construction.
    Transform(const Transform<Scalar, 2>&);

    /* Operators */
    Transform<Scalar, 2>& operator=(const Transform<Scalar, 2>&);
    bool                  operator==(const Transform<Scalar, 2>&) const;
    bool                  operator!=(const Transform<Scalar, 2>&) const;

    /* Get and Set */

    //Get Matrix
    SquareMatrix<Scalar, 2> rotation2x2Matrix() const;
    SquareMatrix<Scalar, 3> rotation3x3Matrix() const;
    SquareMatrix<Scalar, 3> translation3x3Matrix() const;
    SquareMatrix<Scalar, 3> scale3x3Matrix() const;
    SquareMatrix<Scalar, 3> transformMatrix() const;

    //Get Variable
    inline Scalar rotateAngle() const
    {
        return this->rotate_angle_;
    }
    inline Vector<Scalar, 2> translation() const
    {
        return this->translation_;
    }
    inline Vector<Scalar, 2> scale() const
    {
        return this->scale_;
    }

    //Set
    inline void setTranslation(const Vector<Scalar, 2>& translation)
    {
        this->translation_ = translation;
    }
    inline void setRotationAngle(const Scalar& rotate_angle)
    {
        this->rotate_angle_ = rotate_angle;
    }
    inline void setScale(const Vector<Scalar, 2>& scale)
    {
        this->scale_ = scale;
    }
    inline void setRotation(const SquareMatrix<Scalar, 2>& rotation)
    {
        this->rotate_angle_ = acos(rotation(0, 0));
    }
    inline void setRotation(const SquareMatrix<Scalar, 3>& rotation)
    {
        this->rotate_angle_ = acos(rotation(0, 0));
    }
    inline void setIdentity()
    {
        this->rotate_angle_ = 0;
        this->translation_  = Vector<Scalar, 2>(0);
        this->scale_        = Vector<Scalar, 2>(1);
    }

    /* Functions */
    Vector<Scalar, 2> transform(const Vector<Scalar, 2>& input) const;

    Vector<Scalar, 2> rotate(const Vector<Scalar, 2>& input) const;  //there is no class writed for rotate2D, if you want rotate, use this. or use matrix yourself!
    Vector<Scalar, 2> translate(const Vector<Scalar, 2>& input) const;
    Vector<Scalar, 2> scaling(const Vector<Scalar, 2>& input) const;

    static inline Transform<Scalar, 2> identityTransform()
    {
        return Transform<Scalar, 2>();
    }

protected:
    Vector<Scalar, 2> translation_;
    //A rotate_angle can represent the rotation in 2D simplely
    //anticlockwise is + or it is -;
    //Use radian
    Scalar            rotate_angle_;
    Vector<Scalar, 2> scale_;
};

}  //end of namespace PhysIKA

#endif  //PHSYIKA_CORE_TRANSFORM_TRANSFORM_2D_H_
