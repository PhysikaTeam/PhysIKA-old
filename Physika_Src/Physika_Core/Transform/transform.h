/*
 * @file transform.h 
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

#ifndef PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_
#define PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Quaternion/quaternion.h"

namespace Physika{

template <typename Scalar>
class Transform
{
public:
    /* Constructions */
    explicit Transform(const Vector<Scalar,3> );
    explicit Transform(const Quaternion<Scalar> );
    Transform(const Vector<Scalar,3>&, const Quaternion<Scalar>& );
    Transform(const Quaternion<Scalar>&, const Vector<Scalar,3>& );

    /* Get and Set */
    inline Quaternion<Scalar> orientation() const { return orientation_; }
    inline Vector<Scalar,3> position() const { return position_; }
    inline void set_orientation(Quaternion<Scalar> orientation) { orientation_ = orientation; }
    inline void set_position(Vector<Scalar,3> position) { position_ = position; }

protected:
    Quaternion<Scalar> orientation_;
    Vector<Scalar,3> position_;

};

//convenient typedefs
typedef Transform<float> Transformf;
typedef Transform<double> Transformd;

}//end of namespace Physika

#endif //PHSYIKA_CORE_TRANSFORM_TRANSFORM_H_
