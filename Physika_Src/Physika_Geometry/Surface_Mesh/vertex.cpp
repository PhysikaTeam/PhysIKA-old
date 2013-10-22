/*
 * @file vertex.cpp 
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

#include "Physika_Geometry/Surface_Mesh/vertex.h"

namespace Physika{
	
Vertex::Vertex(float x, float y, float z):Vector3f(x,y,z),normal(0)
{

}

Vertex::Vertex(const Vector3f& pos):Vector3f(pos),normal(0)
{

}

} //end of namespace Physika
