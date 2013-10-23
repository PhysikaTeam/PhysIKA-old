/*
 * @file triangle.cpp 
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

#include "Physika_Geometry/Surface_Mesh/triangle.h"
#include "Physika_Geometry/Surface_Mesh/vertex.h"

namespace Physika{

template <typename Scalar>
Triangle<Scalar>::Triangle():normal(0,0,0),
    area(0),
    center(0,0,0)
{
	for(int i = 0;i < 3; ++i)
	{
		edges[i] = NULL;
		vertices[i] = NULL;
	}
}

template <typename Scalar>
Vector3D<Scalar> Triangle<Scalar>::compute_normal()
{
    assert(vertices[0]!=NULL && vertices[1]!=NULL && vertices[2]!=NULL);
    normal = -(vertices[1]->position - vertices[0]->position).cross(vertices[2]->position - vertices[1]->position);
    return normal;
}

template class Triangle<float>;
template class Triangle<double>;

} //end of namespace Physika
