/*
 * @file obj_mesh_io.cpp 
 * @brief load and save mesh to an obj file.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_IO/Mesh_IO/obj_mesh_io.h"

namespace Physika{

template <typename Scalar>
ObjMeshIO<Scalar>::ObjMeshIO()
{
	//int a;
}

template <typename Scalar>
ObjMeshIO<Scalar>::~ObjMeshIO()
{

}

template <typename Scalar>
void ObjMeshIO<Scalar>::load(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

template <typename Scalar>
void ObjMeshIO<Scalar>::save(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

} //end of namespace Physika















