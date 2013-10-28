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

template <typename Scalar>
Vertex<Scalar>::Vertex(Scalar x, Scalar y, Scalar z):position_(x,y,z),normal_(0,0,0)
{
	
}

template <typename Scalar>
Vertex<Scalar>::Vertex(const Vector3D<Scalar>& pos):position_(pos),normal_(0,0,0)
{

}

//explicit instantiation
template class Vertex<float>;
template class Vertex<double>;

} //end of namespace Physika
