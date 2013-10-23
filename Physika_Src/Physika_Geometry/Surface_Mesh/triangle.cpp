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
	
Triangle::Triangle():normal(0)
{

}

Vector3f Triangle::compute_normal()
{
    assert(vertices[0]!=NULL && vertices[1]!=NULL && vertices[2]!=NULL);
    normal = -(*vertices[1] - *vertices[0]).cross(*vertices[2] - *vertices[1]);
    return normal;
}


} //end of namespace Physika
