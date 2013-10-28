/*
 * @file edge.h 
 * @Basic edge class.
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_EDGE_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_EDGE_H_
                  
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar> class Vertex;
template <typename Scalar> class Triangle;

template <typename Scalar>
class Edge
{
public:
    /* Constructions */
    Edge();

    /* Get and Set */
    inline Vector<Scalar,3>& normal() { return normal_; }
    Vertex<Scalar> * vertices(unsigned int i) { return vertices_[i]; }
    Triangle<Scalar> * triangles(unsigned int i){ return triangles_[i]; }

    inline void set_normal(const Vector<Scalar,3>& normal) { normal_ = normal; }
 
protected:
    Vertex<Scalar>* vertices_[2];
	Triangle<Scalar>* triangles_[2];
    Vector<Scalar,3> normal_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_EDGE_H_

