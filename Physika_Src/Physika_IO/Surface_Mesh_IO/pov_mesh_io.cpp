/*
 * @file obj_mesh_io.cpp 
 * @brief load and save mesh to a script file of mesh object for PovRay.
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"
#include "Physika_Core/Utilities/File_Utilities/parse_line.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/pov_mesh_io.h"

namespace Physika{

template <typename Scalar>
bool PovMeshIO<Scalar>::load(const std::string& filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
    return false;
}

template <typename Scalar>
bool PovMeshIO<Scalar>::save(const std::string& filename, const SurfaceMesh<Scalar> *mesh)
{
	std::string file_extension = FileUtilities::fileExtension(filename);
	if(file_extension.size() == 0)
	{
		std::cerr<<"No file extension found for the povmesh file:"<<filename<<std::endl;
		return false;
	}
	if(file_extension != std::string(".povmesh"))
	{
		std::cerr<<"Unknown file format:"<<file_extension<<std::endl;
		return false;
	}
    if(mesh == NULL)
    {
        std::cerr<<"NULL mesh passed to PovMeshIO"<<std::endl;
        return false;
    }
    if(mesh->isTriangularMesh() == false)
    {
        std::cerr<<"PovMeshIO only supports triangle mesh"<<std::endl;
        return false;
    }
	std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
	if(!fileout.is_open())
	{
		std::cerr<<"error in opening file!"<<std::endl;
		return false;
	}
	fileout<<"mesh2 {\n";

	//vertex_vectors
    unsigned int vert_num = mesh->numVertices();
    if(vert_num > 0)
    {
        fileout<<"   vertex_vectors {\n";
        fileout<<"      "<<vert_num<<",\n";
        for (unsigned int idx = 0; idx < vert_num; idx++)
        {
            Vector<Scalar,3> pos = mesh->vertexPosition(idx);
            fileout<<"      <"<<pos[0]<<","<<pos[1]<<","<<-pos[2]<<">"; //negative axis-z due to different coordinate system with OpenGL
            if (idx != vert_num - 1)
                fileout<<",\n";
            else
                fileout<<"\n";
        }
        fileout<<"   }\n"; //end of vertex_vertors
    }

	// normal_vectors
    unsigned int normal_num = mesh->numNormals();
    if(normal_num > 0)
    {
        fileout<<"   normal_vectors {\n";
        fileout<<"      "<<normal_num<<",\n";
        for (unsigned int idx = 0; idx < normal_num; idx++)
        {
            Vector<Scalar,3> normal = mesh->vertexNormal(idx);
            fileout<<"      <"<<normal[0]<<","<<normal[1]<<","<<-normal[2]<<">";//negative axis-z due to different coordinate system with OpenGL
            if (idx != normal_num-1)
                fileout<<",\n";
            else
                fileout<<"\n";
        }
        fileout<<"   }\n"; //end of normal_vertors
    }

	// uv_vectors
    unsigned int tex_coord_num = mesh->numTextureCoordinates();
    if(tex_coord_num > 0)
    {
        fileout<<"   uv_vectors {\n";
        fileout<<"      "<<tex_coord_num<<",\n";
        for (unsigned int idx = 0; idx < tex_coord_num; idx++)
        {
            Vector<Scalar,2> tex_coord = mesh->vertexTextureCoordinate(idx);
            fileout<<"      <"<<tex_coord[0]<<","<<tex_coord[1]<<">";
            if (idx != tex_coord_num - 1)
                fileout<<",\n";
            else
                fileout<<"\n";
        }
        fileout<<"   }\n"; //end of uv_vectors
    }

	// texture_list
    unsigned int tex_num = mesh->numTextures();
    if(tex_num > 0)
    {
        fileout<<"   texture_list {\n";
        fileout<<"      "<<tex_num<<",\n";
        for (unsigned int idx = 0; idx<mesh->numMaterials(); idx++)
        {
            if(mesh->material(idx).hasTexture())
            {
                std::string texture_file_name = mesh->material(idx).textureFileName();
                std::string file_extension = FileUtilities::fileExtension(texture_file_name);
                fileout<<"      "<<"texture{ pigment{ image_map{ "<<file_extension<<" \""<<texture_file_name<<" \"}}}\n";
            }
        }
        fileout<<"   }\n"; // end of texture_list
    }

	
	// face_indices
    unsigned int face_num = mesh->numFaces();
    unsigned int tex_idx = 0;
    if(face_num > 0 && vert_num > 0)
    {
        fileout<<"   face_indices {\n";
        fileout<<"      "<<face_num<<",\n";
        for(unsigned int group_idx = 0; group_idx < mesh->numGroups(); ++group_idx)
        {
            const SurfaceMeshInternal::FaceGroup<Scalar> &group = mesh->group(group_idx);
            const BoundaryMeshInternal::Material<Scalar> &material = mesh->material(group.materialIndex());
            for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
            {
                const SurfaceMeshInternal::Face<Scalar> &face = group.face(face_idx);
                const Vertex<Scalar> & ver0 = face.vertex(0);
                const Vertex<Scalar> & ver1 = face.vertex(1);
                const Vertex<Scalar> & ver2 = face.vertex(2);
                fileout<<"      <"<<ver0.positionIndex()<<","<<ver1.positionIndex()<<","<<ver2.positionIndex()<<">";
                if(material.hasTexture())
                    fileout<<","<<tex_idx;
                if (face_idx != group.numFaces() - 1 || group_idx != mesh->numGroups() - 1)
                    fileout<<",\n";
                else
                    fileout<<"\n";
            }
            if(material.hasTexture())
                ++ tex_idx;
        }
        fileout<<"   }\n"; //end of face_indices
    }

	// normal_indices
    if(face_num > 0 && normal_num > 0)
    {
        fileout<<"   normal_indices {\n";
        fileout<<"      "<<face_num<<",\n";
        for (unsigned int idx = 0; idx < face_num; idx++)
        {
            const SurfaceMeshInternal::Face<Scalar> & face = mesh->face(idx);
            const Vertex<Scalar> & ver0 = face.vertex(0);
            const Vertex<Scalar> & ver1 = face.vertex(1);
            const Vertex<Scalar> & ver2 = face.vertex(2);
            fileout<<"      <"<<ver0.normalIndex()<<","<<ver1.normalIndex()<<","<<ver2.normalIndex()<<">";
            if (idx != face_num - 1)
                fileout<<",\n";
            else
                fileout<<"\n";
        }
        fileout<<"   }\n"; //end of normal_indices
    }

	// uv_indices
    if(face_num > 0 && tex_coord_num > 0)
    {
        fileout<<"   uv_indices {\n";
        fileout<<"      "<<face_num<<",\n";
        for (unsigned int idx = 0; idx < face_num; idx++)
        {
            const SurfaceMeshInternal::Face<Scalar> & face = mesh->face(idx);
            const Vertex<Scalar> & ver0 = face.vertex(0);
            const Vertex<Scalar> & ver1 = face.vertex(1);
            const Vertex<Scalar> & ver2 = face.vertex(2);
            fileout<<"      <"<<ver0.textureCoordinateIndex()<<","<<ver1.textureCoordinateIndex()<<","<<ver2.textureCoordinateIndex()<<">";
            if (idx != face_num - 1)
                fileout<<",\n";
            else
                fileout<<"\n";
        }
        fileout<<"   }\n"; //end of uv_indices
    }

	fileout<<"}\n"; //end of mesh2

	fileout.close();
	return true;
}

//explicit instantitation
template class PovMeshIO<float>;
template class PovMeshIO<double>;

} //end of namespace Physika
