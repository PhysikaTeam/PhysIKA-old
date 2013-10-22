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

using Physika::Type::Vector3f;


namespace Physika{
class Vertex:public Vector3f
{
public:

	Vertex(float x, float y, float z);
	Vertex(const Vector3f& pos);
	Vector3f normal;

protected:
};
} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_

