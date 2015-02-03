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
#include <map>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"
#include "Physika_Core/Utilities/File_Utilities/file_content_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/pov_mesh_io.h"

using Physika::SurfaceMeshInternal::Face;
using Physika::SurfaceMeshInternal::FaceGroup;
using Physika::BoundaryMeshInternal::Material;
using Physika::BoundaryMeshInternal::Vertex;

namespace Physika{

template <typename Scalar>
bool PovMeshIO<Scalar>::load(const std::string& filename, SurfaceMesh<Scalar> *mesh)
{
    if(SurfaceMeshIO<Scalar>::checkFileNameAndMesh(filename,std::string(".povmesh"),mesh) == false)
        return false;
    std::fstream filein(filename.c_str(),std::ios::in);
    if(!filein)
    {
        std::cerr<<"Error: failed to open "<<filename<<"\n";
        return false;
    }
    std::string line;
    unsigned int line_num = 0;
    std::stringstream stream;
    std::string cur_session("UNSET");
    unsigned int cur_normal_idx = 0, cur_uv_idx = 0;
    bool has_texture = false, default_group_created = false;
    while(!filein.eof())
    {
        ++line_num;
        std::getline(filein,line);
        line = FileUtilities::removeWhitespaces(line);
        if(line == std::string("vertex_vectors{"))
            cur_session = std::string("vertex_vectors");
        else if(line == std::string("normal_vectors{"))
            cur_session = std::string("normal_vectors");
        else if(line == std::string("uv_vectors{"))
            cur_session = std::string("uv_vectors");
        else if(line == std::string("texture_list{"))
        {
            cur_session = std::string("texture_list");
            has_texture = true;
        }
        else if(line == std::string("face_indices{"))
            cur_session = std::string("face_indices");
        else if(line == std::string("normal_indices{"))
            cur_session = std::string("normal_indices");
        else if(line == std::string("uv_indices{"))
            cur_session = std::string("uv_indices");
        else
        {
            if(cur_session == std::string("vertex_vectors")
               ||cur_session == std::string("normal_vectors"))
            {
                if(line[0] == '<')
                {
                    std::size_t first_comma_pos = line.find(',');
                    if(first_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::string buffer;
                    Scalar x,y,z;
                    buffer = line.substr(1,first_comma_pos);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>x;
                    std::size_t second_comma_pos = line.find(',',first_comma_pos+1);
                    if(second_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(first_comma_pos+1,second_comma_pos-first_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>y;
                    std::size_t end_pos = line.find('>',second_comma_pos+1);
                    if(end_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(second_comma_pos+1,end_pos-second_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>z;
                    //negative z axis
                    if(cur_session == std::string("vertex_vectors"))
                        mesh->addVertexPosition(Vector<Scalar,3>(x,y,-z));
                    else
                        mesh->addVertexNormal(Vector<Scalar,3>(x,y,-z));
                }
            }
            else if(cur_session == std::string("uv_vectors"))
            {
                if(line[0] == '<')
                {
                    std::size_t first_comma_pos = line.find(',');
                    if(first_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::string buffer;
                    Scalar x,y;
                    buffer = line.substr(1,first_comma_pos);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>x;
                    std::size_t end_pos = line.find('>',first_comma_pos+1);
                    if(end_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(first_comma_pos+1,end_pos-first_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>y;
                    mesh->addVertexTextureCoordinate(Vector<Scalar,2>(x,y));
                }
            }
            else if(cur_session == std::string("texture_list"))
            {
                if(line.substr(0,8) == std::string("texture{"))
                {
                    std::size_t first_quotation_pos = line.find(std::string("\""));
                    if(first_quotation_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::size_t second_quotation_pos = line.find(std::string("\""),first_quotation_pos+1);
                    if(second_quotation_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::string tex_file_name = line.substr(first_quotation_pos+1,second_quotation_pos-first_quotation_pos-1);
                    //as material information is lost in .povmesh, we bind one predefind material with each texture
                    //create one group for each material
                    FaceGroup<Scalar> group(tex_file_name);
                    mesh->addGroup(group);
                    Material<Scalar> material = Material<Scalar>::Iron();
                    tex_file_name = FileUtilities::dirName(filename) + "/" + tex_file_name;
                    material.setTextureFileName(tex_file_name);
                    mesh->addMaterial(material);
                }
            }
            else if(cur_session == std::string("face_indices"))
            {
                if(line[0] == '<')
                {
                    //first read the face information
                    std::size_t first_comma_pos = line.find(',');
                    if(first_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::string buffer;
                    unsigned int x,y,z,tex_idx;
                    buffer = line.substr(1,first_comma_pos);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>x;
                    std::size_t second_comma_pos = line.find(',',first_comma_pos+1);
                    if(second_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(first_comma_pos+1,second_comma_pos-first_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>y;
                    std::size_t end_pos = line.find('>',second_comma_pos+1);
                    if(end_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(second_comma_pos+1,end_pos-second_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>z;
                    Vector<unsigned int,3> indices(x,y,z);
                    if(has_texture)
                    {
                        std::size_t comma_before_tex_pos = line.find(',',end_pos+1);
                        if(comma_before_tex_pos == std::string::npos)
                        {
                            std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                            return false;
                        }
                        std::size_t comma_after_tex_pos = line.find(',',comma_before_tex_pos+1);
                        if(comma_after_tex_pos == std::string::npos) //last face
                            buffer = line.substr(comma_before_tex_pos+1);
                        else
                            buffer = line.substr(comma_before_tex_pos+1,comma_after_tex_pos-comma_before_tex_pos-1);
                        stream.str(std::string());
                        stream.clear();
                        stream<<buffer;
                        stream>>tex_idx;
                    }
                    else
                    {
                        if(!default_group_created) //create default group if not created yet
                        {
                            FaceGroup<Scalar> group(std::string("default"),0);
                            mesh->addGroup(group);
                            mesh->addMaterial(Material<Scalar>::Iron());
                            default_group_created = true;
                        }
                        tex_idx = 0;
                    }
                    FaceGroup<Scalar> &group = mesh->group(tex_idx);
                    Face<Scalar> face;
                    for(unsigned int vert_idx = 0; vert_idx < 3; ++vert_idx)
                    {
                        Vertex<Scalar> vert(indices[vert_idx]);
                        face.addVertex(vert);
                    }
                    group.addFace(face);
                }
            }
            else if(cur_session == std::string("normal_indices")
                    ||cur_session == std::string("uv_indices"))
            {
                if(line[0] == '<')
                {
                    //first read the normal/uv index of the face's three vertices
                    std::size_t first_comma_pos = line.find(',');
                    if(first_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    std::string buffer;
                    unsigned int x,y,z;
                    buffer = line.substr(1,first_comma_pos);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>x;
                    std::size_t second_comma_pos = line.find(',',first_comma_pos+1);
                    if(second_comma_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(first_comma_pos+1,second_comma_pos-first_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>y;
                    std::size_t end_pos = line.find('>',second_comma_pos+1);
                    if(end_pos == std::string::npos)
                    {
                        std::cerr<<"Error: invalid line: "<<line_num<<"\n";
                        return false;
                    }
                    buffer = line.substr(second_comma_pos+1,end_pos-second_comma_pos-1);
                    stream.str(std::string());
                    stream.clear();
                    stream<<buffer;
                    stream>>z;
                    Vector<unsigned int,3> indices(x,y,z);
                    //now set the data into mesh
                    unsigned int &cur_idx = cur_session == std::string("normal_indices") ? cur_normal_idx : cur_uv_idx;
                    Face<Scalar> &face = mesh->face(cur_idx); //this face should have been created
                    for(unsigned int vert_idx = 0; vert_idx < 3; ++vert_idx)
                    {
                        Vertex<Scalar> &vert = face.vertex(vert_idx);
                        if(cur_session == std::string("normal_indices"))
                            vert.setNormalIndex(indices[vert_idx]);
                        else
                            vert.setTextureCoordinateIndex(indices[vert_idx]);
                    }
                    ++cur_idx;
                }
            }
            else if(cur_session == std::string("UNSET"))
            {
                //do nothing
            }
        }
    }
    filein.close();
    return true;
}

template <typename Scalar>
bool PovMeshIO<Scalar>::save(const std::string& filename, const SurfaceMesh<Scalar> *mesh)
{
    if(SurfaceMeshIO<Scalar>::checkFileNameAndMesh(filename,std::string(".povmesh"),mesh) == false)
        return false;
    if(mesh->isTriangularMesh() == false)
    {
        std::cerr<<"PovMeshIO only supports triangle mesh\n";
        return false;
    }
	std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
	if(!fileout.is_open())
	{
		std::cerr<<"Error: failed to open "<<filename<<"\n";
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
    std::map<std::string,unsigned int> texture_name_idx_map;
    unsigned int tex_idx = 0;
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
                fileout<<"      "<<"texture{ pigment{ image_map{ "<<file_extension<<" \""<<FileUtilities::filenameInPath(texture_file_name)<<" \"}}}\n";
                if(texture_name_idx_map.find(texture_file_name) == texture_name_idx_map.end())
                {
                    texture_name_idx_map.insert(std::make_pair(texture_file_name,tex_idx));
                    ++tex_idx;
                }
            }
        }
        fileout<<"   }\n"; // end of texture_list
    }

    	
	// face_indices
    unsigned int face_num = mesh->numFaces();
    if(face_num > 0 && vert_num > 0)
    {
        fileout<<"   face_indices {\n";
        fileout<<"      "<<face_num<<",\n";
        for(unsigned int group_idx = 0; group_idx < mesh->numGroups(); ++group_idx)
        {
            const SurfaceMeshInternal::FaceGroup<Scalar> &group = mesh->group(group_idx);
            const BoundaryMeshInternal::Material<Scalar> &material = mesh->material(group.materialIndex());
            std::string texture_file_name = material.textureFileName();
            unsigned int tex_idx = 0;
            if(material.hasTexture())
                tex_idx = texture_name_idx_map[texture_file_name];
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
