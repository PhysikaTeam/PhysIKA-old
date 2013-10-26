/*
 * @file edge.cpp 
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

#include "Physika_Geometry/Surface_Mesh/edge.h"
#include "Physika_Geometry/Surface_Mesh/triangle.h"
#include "Physika_Geometry/Surface_Mesh/vertex.h"


namespace Physika{

template <typename Scalar>
Edge<Scalar>::Edge():normal_(0,0,0)
{
	vertices_[0] = NULL;
	vertices_[1] = NULL;
	triangles_[0] = NULL;
	triangles_[1] = NULL;
}



template class Edge<float>;
template class Edge<double>;

} //end of namespace Physika
