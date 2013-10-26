/*
 * @file vertex.h 
 * @Basic vertex class.
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_


#include "Physika_Core/Vectors/vector.h"

using Physika::Vector3D;


namespace Physika{

template <typename Scalar>
class Vertex
{
public:
    /* Constructions*/
    Vertex(Scalar , Scalar , Scalar );
    Vertex(const Vector3D<Scalar>& pos);

    /* Get and Set */
    inline Vector3D<Scalar>& position()  { return position_; }
    inline Vector3D<Scalar>& normal()  { return normal_; }
    inline void set_position(Vector3D<Scalar> position) { position_ = position; }
    inline void set_normal(Vector3D<Scalar> normal) { normal_ = normal; }
    
    /* Protected Members */
protected:
	Vector3D<Scalar> position_;
	Vector3D<Scalar> normal_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_

