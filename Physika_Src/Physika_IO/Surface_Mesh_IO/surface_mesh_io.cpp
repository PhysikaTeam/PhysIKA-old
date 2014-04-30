/*
 * @file surface_mesh_IO.cpp 
 * @brief surface mesh loader/saver, load/save surface mesh from/to file.
 *        dynamically choose different loader/saver with respect to file suffix.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/stl_mesh_io.h"

namespace Physika{

template <typename Scalar>
void SurfaceMeshIO<Scalar>::load(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    string::size_type suffix_idx = filename.find('.');
    PHYSIKA_ASSERT(suffix_idx<filename.size());
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".obj"))
	ObjMeshIO<Scalar>::load(filename,mesh);
    else if(suffix==string(".stl"))
	StlMeshIO<Scalar>::load(filename,mesh);
    else
	PHYSIKA_ERROR("Unknown mesh file format!");
}

template <typename Scalar>
void SurfaceMeshIO<Scalar>::save(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    string::size_type suffix_idx = filename.find('.');
    PHYSIKA_ASSERT(suffix_idx<filename.size());
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".obj"))
	ObjMeshIO<Scalar>::save(filename,mesh);
    else if(suffix==string(".stl"))
	StlMeshIO<Scalar>::save(filename,mesh);
    else
	PHYSIKA_ERROR("Unknown mesh file format specified!");
}

} //end of namespace Physika


















