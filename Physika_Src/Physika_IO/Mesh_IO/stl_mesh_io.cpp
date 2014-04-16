/*
 * @file obj_mesh_io.cpp 
 * @brief load and save mesh to a stl file.
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

#include "Physika_IO/Mesh_IO/stl_mesh_io.h"

namespace Physika{

template <typename Scalar>
StlMeshIO<Scalar>::StlMeshIO()
{
	//int a;
}

template <typename Scalar>
StlMeshIO<Scalar>::~StlMeshIO()
{

}

template <typename Scalar>
static void StlMeshIO<Scalar>::load(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

template <typename Scalar>
static void StlMeshIO<Scalar>::save(const string& filename, SurfaceMesh<Scalar> *mesh)
{

}

} //end of namespace Physika










