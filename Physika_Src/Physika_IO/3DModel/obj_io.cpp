/*
 * @file obj_IO.cpp 
 * @Basic obj_IO, load a mesh or write a mesh to a obj file. simply without texture.
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

#include "Physika_IO/3DModel/obj_io.h"

namespace Physika{

template <typename Scalar>
ObjIO<Scalar>::ObjIO()
{
	//int a;
}

template <typename Scalar>
ObjIO<Scalar>::~ObjIO()
{

}

template <typename Scalar>
void ObjIO<Scalar>::read(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

template <typename Scalar>
void ObjIO<Scalar>::write(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

} //end of namespace Physika
