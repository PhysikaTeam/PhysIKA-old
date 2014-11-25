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
using std::string;

namespace Physika{

template <typename Scalar>
bool PovMeshIO<Scalar>::load(const string& filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
    return false;
}

template <typename Scalar>
bool PovMeshIO<Scalar>::save(const string& filename, const SurfaceMesh<Scalar> *mesh)
{
	string file_extension = FileUtilities::fileExtension(filename);
	if(file_extension.size() == 0)
	{
		std::cerr<<"No file extension found for the povmesh file:"<<filename<<std::endl;
		return false;
	}
	if(file_extension != string(".povmesh"))
	{
		std::cerr<<"Unknown file format:"<<file_extension<<std::endl;
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
	fileout<<"   vertex_vectors {\n";
	fileout<<"      "<<mesh->numVertices()<<",\n";
	for (unsigned int idx=0; idx<mesh->numVertices(); idx++)
	{
		Vector<Scalar,3> pos = mesh->vertexPosition(idx);
		fileout<<"      <"<<pos[0]<<","<<pos[1]<<","<<pos[2]<<">";
		if (idx != mesh->numVertices()-1)
			fileout<<",\n";
		else
			fileout<<"\n";
	}
	fileout<<"   }\n"; //end of vertex_vertors

	// normal_vectors
	fileout<<"   normal_vectors {\n";
	fileout<<"      "<<mesh->numNormals()<<",\n";
	for (unsigned int idx=0; idx<mesh->numNormals(); idx++)
	{
		Vector<Scalar,3> normal = mesh->vertexNormal(idx);
		fileout<<"      <"<<normal[0]<<","<<normal[1]<<","<<normal[2]<<">";
		if (idx != mesh->numNormals()-1)
			fileout<<",\n";
		else
			fileout<<"\n";
	}
	fileout<<"   }\n"; //end of normal_vertors

	// uv_vectors
	fileout<<"   uv_vectors {\n";
	fileout<<"      "<<mesh->numTextureCoordinates()<<",\n";
	for (unsigned int idx = 0; idx<mesh->numTextureCoordinates(); idx++)
	{
		Vector<Scalar,2> tex_coord = mesh->vertexTextureCoordinate(idx);
		fileout<<"      <"<<tex_coord[0]<<","<<tex_coord[1]<<">";
		if (idx != mesh->numTextureCoordinates()-1)
			fileout<<",\n";
		else
			fileout<<"\n";
	}
	fileout<<"   }\n"; //end of uv_vectors

	// texture_list
	fileout<<"   texture_list {\n";
	fileout<<"      "<<mesh->numMaterials()<<",\n";
	for (unsigned int idx = 0; idx<mesh->numMaterials(); idx++)
		fileout<<"      "<<"texture{ pigment{ image_map{ png \""<<mesh->material(idx).textureFileName()<<" \"}}}\n";
	fileout<<"   }\n"; // end of texture_list

	unsigned int num_face = mesh->numFaces();
	// face_indices 
	fileout<<"   face_indices {\n";
	fileout<<"      "<<num_face<<",\n";
	for (unsigned int idx=0; idx<num_face; idx++)
	{
		const SurfaceMeshInternal::Face<Scalar> & face = mesh->face(idx);
		const Vertex<Scalar> & ver0 = face.vertex(0);
		const Vertex<Scalar> & ver1 = face.vertex(1);
		const Vertex<Scalar> & ver2 = face.vertex(2);
		fileout<<"      <"<<ver0.positionIndex()<<","<<ver1.positionIndex()<<","<<ver2.positionIndex()<<">";
		if (idx != num_face-1)
			fileout<<",\n";
		else
			fileout<<"\n";
	}
	fileout<<"   }\n"; //end of face_indices

	// normal_indices
	fileout<<"   normal_indices {\n";
	fileout<<"      "<<num_face<<",\n";
	for (unsigned int idx=0; idx<num_face; idx++)
	{
		const SurfaceMeshInternal::Face<Scalar> & face = mesh->face(idx);
		const Vertex<Scalar> & ver0 = face.vertex(0);
		const Vertex<Scalar> & ver1 = face.vertex(1);
		const Vertex<Scalar> & ver2 = face.vertex(2);
		fileout<<"      <"<<ver0.normalIndex()<<","<<ver1.normalIndex()<<","<<ver2.normalIndex()<<">";
		if (idx != num_face-1)
			fileout<<",\n";
		else
			fileout<<"\n";
	}
	fileout<<"   }\n"; //end of normal_indices

	// uv_indices
	fileout<<"   uv_indices {\n";
	fileout<<"      "<<num_face<<",\n";
	for (unsigned int group_idx = 0; group_idx<mesh->numGroups(); group_idx++)
	{
		const SurfaceMeshInternal::FaceGroup<Scalar> & group = mesh->group(group_idx);
		for (unsigned int face_idx = 0; face_idx<group.numFaces(); face_idx++)
		{
			const SurfaceMeshInternal::Face<Scalar> & face = group.face(face_idx);
			const Vertex<Scalar> & ver0 = face.vertex(0);
			const Vertex<Scalar> & ver1 = face.vertex(1);
			const Vertex<Scalar> & ver2 = face.vertex(2);
			fileout<<"      <"<<ver0.textureCoordinateIndex()<<","<<ver1.textureCoordinateIndex()<<","<<ver2.textureCoordinateIndex()<<">,"<<group.materialIndex();
			if (group_idx != mesh->numGroups()-1 || face_idx != group.numFaces()-1)
				fileout<<",\n";
			else
				fileout<<"\n";
		}
	}
	fileout<<"   }\n"; //end of uv_indices

	fileout<<"}\n"; //end of mesh2

	fileout.close();
	return true;
}

//explicit instantitation
template class PovMeshIO<float>;
template class PovMeshIO<double>;

} //end of namespace Physika
