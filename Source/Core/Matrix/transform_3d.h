/*
 * @file transform_3d.h 
 * @brief transform3d class, brief class representing a rigid 3d transform use matrix,vector and quaternion
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

#ifndef PHSYIKA_CORE_TRANSFORM_TRANSFORM_3D_H_
#define PHSYIKA_CORE_TRANSFORM_TRANSFORM_3D_H_

#include "../Vector.h"
#include "../Matrix.h"
#include "../Utility.h"
#include "../Quaternion/quaternion.h"
#include "transform.h"

namespace PhysIKA {

/*
 * Transform is defined for float and double
 */

template <typename Scalar>
class Transform<Scalar, 3>
{
public:
    /* Constructions */
    Transform();
    ~Transform();
    explicit Transform(const Vector<Scalar, 3>& translation);
    explicit Transform(const Quaternion<Scalar>& rotation);
    Transform(const Vector<Scalar, 3>& translation, const Quaternion<Scalar>& rotation);
    Transform(const Vector<Scalar, 3>& translation, const Quaternion<Scalar>& rotation, const Vector<Scalar, 3>& scale);
    Transform(const SquareMatrix<Scalar, 4>&);  //now it's not all right.Suggest not use this construction.
    Transform(const SquareMatrix<Scalar, 3>&);
    Transform(const Transform<Scalar, 3>&);

    /*operator*/
    Transform<Scalar, 3>& operator=(const Transform<Scalar, 3>&);
    bool                  operator==(const Transform<Scalar, 3>&) const;
    bool                  operator!=(const Transform<Scalar, 3>&) const;

    /* Get and Set */

    //Get Matrix;
    SquareMatrix<Scalar, 3> rotation3x3Matrix() const;
    SquareMatrix<Scalar, 4> rotation4x4Matrix() const;
    SquareMatrix<Scalar, 4> translation4x4Matrix() const;
    SquareMatrix<Scalar, 4> scale4x4Matrix() const;
    SquareMatrix<Scalar, 4> transformMatrix() const;

    //Get Variable
    inline Quaternion<Scalar> rotation() const
    {
        return rotation_;
    }
    inline Vector<Scalar, 3> translation() const
    {
        return translation_;
    }
    inline Vector<Scalar, 3> scale() const
    {
        return scale_;
    }

    //Set
    inline void setRotation(const Vector<Scalar, 3>& unit_axis, Scalar angle_rad)
    {
        rotation_ = Quaternion<Scalar>(unit_axis, angle_rad);
    }
    inline void setRotation(const Quaternion<Scalar>& rotation)
    {
        rotation_ = rotation;
    }
    inline void setRotation(const SquareMatrix<Scalar, 3>& rotation)
    {
        rotation_ = Quaternion<Scalar>(rotation);
    }
    inline void setRotation(const Vector<Scalar, 3>& rotation)
    {
        rotation_ = Quaternion<Scalar>(rotation);
    }

    inline void setScale(const Vector<Scalar, 3>& scale)
    {
        scale_ = scale;
    }
    inline void setUniformScalar(Scalar scale)
    {
        scale_ = Vector<Scalar, 3>(scale);
    }
    inline void setTranslation(const Vector<Scalar, 3>& translation)
    {
        translation_ = translation;
    }
    inline void setIdentity()
    {
        rotation_    = Quaternion<Scalar>(0, 0, 0, 1);
        translation_ = Vector<Scalar, 3>(0, 0, 0);
        scale_       = Vector<Scalar, 3>(1, 1, 1);
    }

    /* Functions*/
    //Order is scale > rotate > translate. If you want another order, you can get scale/rotation/translation component and do it yourself.
    Vector<Scalar, 3> transform(const Vector<Scalar, 3>& input) const;

    Vector<Scalar, 3> rotate(const Vector<Scalar, 3>& input) const;
    Vector<Scalar, 3> translate(const Vector<Scalar, 3>& input) const;
    Vector<Scalar, 3> scaling(const Vector<Scalar, 3>& input) const;

    static inline Transform<Scalar, 3> identityTransform()
    {
        return Transform<Scalar, 3>();
    }

protected:
    Vector<Scalar, 3>  translation_;
    Quaternion<Scalar> rotation_;
    Vector<Scalar, 3>  scale_;
};

using Transform3f = Transform<float, 3>;
using Transform3d = Transform<double, 3>;

}  //end of namespace PhysIKA

#endif  //PHSYIKA_CORE_TRANSFORM_TRANSFORM_3D_H_
