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


#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar>
class Vertex
{
public:
    /* Constructions*/
    Vertex(Scalar , Scalar , Scalar );
    Vertex(const Vector<Scalar,3>& pos);

    /* Get and Set */
    inline Vector<Scalar,3>& position()  { return position_; }
    inline Vector<Scalar,3>& normal()  { return normal_; }
    inline void setPosition(Vector<Scalar,3> position) { position_ = position; }
    inline void setNormal(Vector<Scalar,3> normal) { normal_ = normal; }
    
    /* Protected Members */
protected:
	Vector<Scalar,3> position_;
	Vector<Scalar,3> normal_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_

