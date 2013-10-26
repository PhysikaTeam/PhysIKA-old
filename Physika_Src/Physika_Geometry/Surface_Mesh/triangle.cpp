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
Triangle<Scalar>::Triangle():normal_(0,0,0),
    area_(0),
    center_(0,0,0)
{
	for(int i = 0;i < 3; ++i)
	{
		edges_[i] = NULL;
		vertices_[i] = NULL;
	}
}

template <typename Scalar>
Vector3D<Scalar> Triangle<Scalar>::computeNormals()
{
    assert(vertices_[0]!=NULL && vertices_[1]!=NULL && vertices_[2]!=NULL);
    normal_ = -(vertices_[1]->position() - vertices_[0]->position()).cross(vertices_[2]->position() - vertices_[1]->position());
    return normal_;
}

template class Triangle<float>;
template class Triangle<double>;

} //end of namespace Physika
