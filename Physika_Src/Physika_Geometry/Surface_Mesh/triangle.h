/*
 * @file triangle.h 
 * @Basic triangle class.
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_TRIANGLE_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_TRIANGLE_H_


#include "Physika_Core/Vectors/vector.h"


using Physika::Type::Vector3f;


namespace Physika{

class Edge;
class Vertex;

class Triangle
{
public:
    Triangle();
    Vector3f compute_normal();

    Vertex *vertices[3];
    Edge *edges[3];
    Vector3f normal;
    Vector3f center;
    float area;


protected:
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_TRIANGLE_H_

