/*
 * @file surface_mesh_IO.cpp 
 * @brief surface mesh loader/saver, load/save surface mesh from/to file.
 *        dynamically choose different loader/saver with respect to file suffix.
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

#include <iostream>
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/stl_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/pov_mesh_io.h"

using std::string;

namespace Physika{

template <typename Scalar>
bool SurfaceMeshIO<Scalar>::load(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    string suffix = FileUtilities::fileExtension(filename);
    if(suffix==string(".obj"))
        return ObjMeshIO<Scalar>::load(filename,mesh);
    else if(suffix==string(".stl"))
        return StlMeshIO<Scalar>::load(filename,mesh);
    else if(suffix==string(".povmesh"))
        return PovMeshIO<Scalar>::load(filename,mesh);
    else
    {
        std::cerr<<"Unknown mesh file format: "<<suffix<<"!\n";
        return false;
    }
}

template <typename Scalar>
bool SurfaceMeshIO<Scalar>::save(const string &filename, const SurfaceMesh<Scalar> *mesh)
{
    string suffix = FileUtilities::fileExtension(filename);
    if(suffix==string(".obj"))
        return ObjMeshIO<Scalar>::save(filename,mesh);
    else if(suffix==string(".stl"))
        return StlMeshIO<Scalar>::save(filename,mesh);
    else if(suffix==string(".povmesh"))
        return PovMeshIO<Scalar>::save(filename,mesh);
    else
    {
        std::cerr<<"Unknown mesh file format specified: "<<suffix<<"!\n";
        return false;
    }
}

template <typename Scalar>
bool SurfaceMeshIO<Scalar>::checkFileNameAndMesh(const std::string &filename, const std::string &expected_extension, const SurfaceMesh<Scalar> *mesh)
{
	std::string file_extension = FileUtilities::fileExtension(filename);
	if(file_extension.size() == 0)
	{
		std::cerr<<"No file extension found for the mesh file:"<<filename<<std::endl;
		return false;
	}
	if(file_extension != expected_extension)
	{
		std::cerr<<"Unknown file format:"<<file_extension<<std::endl;
		return false;
	}
    if(mesh == NULL)
    {
        std::cerr<<"NULL mesh passed to MeshIO"<<std::endl;
        return false;
    }
    return true;
}

// //explicit instantitation
template class SurfaceMeshIO<float>;
template class SurfaceMeshIO<double>;

} //end of namespace Physika
