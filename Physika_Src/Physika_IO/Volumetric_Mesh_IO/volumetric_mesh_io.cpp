/*
 * @file volumetric_mesh_io.cpp 
 * @brief volumetric mesh loader/saver, load/save volumetric mesh from/to file.
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

#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>* VolumetricMeshIO<Scalar,Dim>::load(const std::string &filename)
{
    return NULL;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::save(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh)
{
    return false;
}

//explicit instantiations
template class VolumetricMeshIO<float,2>;
template class VolumetricMeshIO<float,3>;
template class VolumetricMeshIO<double,2>;
template class VolumetricMeshIO<double,3>;

}  //end of namespace Physika


















