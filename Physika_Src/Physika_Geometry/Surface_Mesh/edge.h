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


#include "Physika_Core/Vectors/vector.h"

using Physika::Vector3D;
                  

namespace Physika{

template <typename Scalar> class Vertex;
template <typename Scalar> class Triangle;

template <typename Scalar>
class Edge
{
public:
    Edge();

    Vertex<Scalar> *vertices[2];
	Triangle<Scalar> *triangles[2];
    Vector3D<Scalar> normal;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_EDGE_H_

