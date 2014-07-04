/*
 * @file transform.h 
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

#ifndef PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_
#define PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

namespace Physika{

/*
 * Transform is defined for float and double
 */

template <typename Scalar>
class Transform
{
public:
    /* Constructions */
    Transform();
    explicit Transform(const Vector<Scalar,3> );
    explicit Transform(const Quaternion<Scalar> );
    Transform(const Vector<Scalar,3>&, const Quaternion<Scalar>& );
    Transform(const Vector<Scalar,3>&, const Quaternion<Scalar>&, const Vector<Scalar, 3>& );
    Transform(const Quaternion<Scalar>&, const Vector<Scalar,3>& );
    Transform(const Quaternion<Scalar>&, const Vector<Scalar,3>&, const Vector<Scalar, 3>& );
    Transform(const SquareMatrix<Scalar,4>& );
    Transform(const SquareMatrix<Scalar,3>&);

    /* Get and Set */
    inline Quaternion<Scalar> rotation() const { return rotation_; }
    inline SquareMatrix<Scalar, 3> rotation3x3Matrix() const { return rotation_.get3x3Matrix(); }
    inline SquareMatrix<Scalar,4> rotation4x4Matrix() const { return rotation_.get4x4Matrix(); }
    inline SquareMatrix<Scalar,4> transformMatrix() const
    {
        SquareMatrix<Scalar,4> matrix = rotation_.get4x4Matrix();
        matrix(0, 3) = translation_[0];
        matrix(1, 3) = translation_[1];
        matrix(2, 3) = translation_[2];
        return matrix;
    }
    inline Vector<Scalar, 3> translation() const { return translation_; }
    inline Vector<Scalar, 3> scale() const { return scale_; }
    inline void setRotation(const Quaternion<Scalar>& rotation) { rotation_ = rotation; }
    inline void setRotation(const SquareMatrix<Scalar, 3>& rotation) { rotation_ = Quaternion<Scalar>(rotation); }
    inline void setRotation(const Vector<Scalar, 3>& rotation ) { rotation_ = Quaternion<Scalar>(rotation); }
    inline void setScale(const Vector<Scalar, 3> scale ) { scale_ = scale; }
    inline void setTranslation(const Vector<Scalar,3>& translation) { translation_ = translation; }
    inline void setIdentity() { rotation_ = Quaternion<Scalar>(0,0,0,1); translation_ = Vector<Scalar, 3>(0,0,0);}

    /* Funtions*/
    Vector<Scalar,3> transform(const Vector<Scalar, 3>& input) const;

    static inline Transform<Scalar> identityTransform() { return Transform<Scalar>(); }

protected:
    Quaternion<Scalar> rotation_;
    Vector<Scalar,3> translation_;
    Vector<Scalar,3> scale_;
protected:
    PHYSIKA_STATIC_ASSERT((is_same<Scalar,float>::value||is_same<Scalar,double>::value),
                           "Transform<Scalar> are only defined for Scalar type of float and double");
};

//convenient typedefs
typedef Transform<float> Transformf;
typedef Transform<double> Transformd;

}//end of namespace Physika

#endif //PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_
