/*
 * @file obj_mesh_io.cpp 
 * @brief load and save mesh to a stl file.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/stl_mesh_io.h"
using std::string;

namespace Physika{

template <typename Scalar>
bool StlMeshIO<Scalar>::load(const string& filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
    return false;
}

template <typename Scalar>
bool StlMeshIO<Scalar>::save(const string& filename, const SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
    return false;
}

//explicit instantitation
template class StlMeshIO<float>;
template class StlMeshIO<double>;

} //end of namespace Physika
