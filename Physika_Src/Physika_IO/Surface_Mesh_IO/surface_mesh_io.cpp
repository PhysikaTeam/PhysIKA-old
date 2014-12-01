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

#include <iostream>
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/stl_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/pov_mesh_io.h"

using std::string;

namespace Physika{

template <typename Scalar>
bool SurfaceMeshIO<Scalar>::load(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension found for the mesh file!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".obj"))
        return ObjMeshIO<Scalar>::load(filename,mesh);
    else if(suffix==string(".stl"))
        return StlMeshIO<Scalar>::load(filename,mesh);
    else if(suffix==string(".povmesh"))
        return PovMeshIO<Scalar>::load(filename,mesh);
    else
    {
        std::cerr<<"Unknown mesh file format!\n";
        return false;
    }
}

template <typename Scalar>
bool SurfaceMeshIO<Scalar>::save(const string &filename, const SurfaceMesh<Scalar> *mesh)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension specified for the mesh file!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".obj"))
        return ObjMeshIO<Scalar>::save(filename,mesh);
    else if(suffix==string(".stl"))
        return StlMeshIO<Scalar>::save(filename,mesh);
    else if(suffix==string(".povmesh"))
        return PovMeshIO<Scalar>::save(filename,mesh);
    else
    {
        std::cerr<<"Unknown mesh file format specified!\n";
        return false;
    }
}

// //explicit instantitation
template class SurfaceMeshIO<float>;
template class SurfaceMeshIO<double>;

} //end of namespace Physika
