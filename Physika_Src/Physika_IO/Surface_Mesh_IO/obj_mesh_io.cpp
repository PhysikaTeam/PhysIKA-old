/*
* @file obj_mesh_io.cpp 
* @brief load and save mesh to an obj file.
* @author Fei Zhu, Liyou Xu
* 
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include <cstdio>
#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"
#include "Physika_Core/Utilities/File_Utilities/file_content_utilities.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"

using Physika::SurfaceMeshInternal::Face;
using Physika::SurfaceMeshInternal::FaceGroup;
using Physika::BoundaryMeshInternal::Material;
using Physika::BoundaryMeshInternal::Vertex;

namespace Physika{
    
template <typename Scalar>
bool ObjMeshIO<Scalar>::load(const std::string &filename, SurfaceMesh<Scalar> *mesh)
{
    if(SurfaceMeshIO<Scalar>::checkFileNameAndMesh(filename,std::string(".obj"),mesh) == false)
        return false;
    // a indicator points to current group 
    FaceGroup<Scalar>* current_group = NULL;
    // num of total faces
    unsigned int num_face = 0;
    // index of the current material in mesh's materials
    unsigned int current_material_index = 0;
    // the line number in the obj file
    unsigned int line_num = 0;
    unsigned int num_group_faces = 0;
    // group name
    std::string group_source_name;
    unsigned int group_clone_index = 0;
    std::fstream ifs( filename.c_str(),std::ios::in);
    if(!ifs)
    {
        std::cerr<<"Error: failed to open "<<filename<<"\n";
        return false;
    }
    std::string  line;
    std::stringstream stream;
    while(!ifs.eof())
    {
        ++line_num;
        std::getline(ifs,line);
        line = FileUtilities::removeWhitespaces(line,1);
        std::string line_next;
        unsigned int line_length=line.size();
        while (line_length > 0 && line[line_length-1] == '\\')     //if the last character in a line is '\',we will merge nextline into this one
        {
            std::getline(ifs,line_next);
            line_next = FileUtilities::removeWhitespaces(line_next,1);
            line[line_length-1] = ' ';
            line = line + line_next;
            line_length = line.size();
        }
        stream.str("");
        stream.clear();
        stream<<line;
        std::string type_of_line;
        stream>>type_of_line;
        if(type_of_line[0] == 'v'&& type_of_line.size() == 1)
        {
            //vertex
            Scalar x,y,z;
            if(!(stream>>x))
            {
                std::cerr<<"Error:stream>>x "<<"line:"<<line_num<<std::endl;
                return false;
            }
            if(!(stream>>y))
            {
                std::cerr<<"Error:stream>>y "<<"line:"<<line_num<<std::endl;
                return false;
            }
            if(!(stream>>z))
            {
                std::cerr<<"Error:stream>>z "<<"line:"<<line_num<<std::endl;
                return false;
            }
            mesh->addVertexPosition(Vector<Scalar,3>(x,y,z));
        }
        else if(type_of_line == std::string("vn"))
        {   //vertex normal
            Scalar x,y,z;
            if(!(stream>>x))
            {
                std::cerr<<"x position of a normal read error "<<"line:"<<line_num<<std::endl;
                return false;
            }
            if(!(stream>>y))
            {
                std::cerr<<"y position of a normal read error "<<"line:"<<line_num<<std::endl;
                return false;
            }
            if(!(stream>>z))
            {
                std::cerr<<"z position of a normal read error "<<"line:"<<line_num<<std::endl;
                return false;
            }
            mesh->addVertexNormal(Vector<Scalar,3>(x,y,z));
        }
        else if(type_of_line == std::string("vt"))
        {   //vertex texture
            Scalar x,y;
            if(!(stream>>x))
            {
                std::cerr<<"x position of a texture read error "<<"line:"<<line_num<<std::endl;
                return false;
            }
            if(!(stream>>y))
            {
                std::cerr<<"y position of a texture read error "<<"line:"<<line_num<<std::endl;
                return false;
            }
            mesh->addVertexTextureCoordinate(Vector<Scalar,2>(x,y));
        }
        else if(type_of_line == std::string("g"))
        {
            std::string group_name;
            stream>>group_name;
            unsigned int length=group_name.size();
            if(length < 1)
            {
                std::cerr<<"Warning: parsed empty group name and we consider it as a default group "<<"line:"<<line_num<<std::endl;
                FaceGroup<Scalar> *p=mesh->groupPtr(std::string("default"));
                if(p == NULL)
                {
                    mesh->addGroup(FaceGroup<Scalar>(std::string("default"),current_material_index));
                    current_group = mesh->groupPtr(std::string("default"));
                    group_source_name = std::string("");
                    group_clone_index=0;
                    num_group_faces=0;
                }
                else current_group = p;
            }
            else if( (current_group = mesh->groupPtr(group_name)) )
            {
            }
            else
            {
                mesh->addGroup(FaceGroup<Scalar>(group_name,current_material_index));
                current_group=mesh->groupPtr(group_name);
                group_source_name=group_name;
                group_clone_index=0;
                num_group_faces=0;
            }

        }
        else if(type_of_line == std::string("f") || type_of_line == std::string("fo"))
        {
            //std::cerr<<line<<std::endl;
            if(current_group==NULL)
            {
                mesh->addGroup(FaceGroup<Scalar>(std::string("default"),current_material_index));
                current_group=mesh->groupPtr(std::string("default"));
            }
            Face<Scalar> face_temple;
            std::string vertex_indice;
            while(stream>>vertex_indice)
            {
                unsigned int pos;
                unsigned int nor;
                unsigned int tex;
                if(vertex_indice.find(std::string("//")) != std::string::npos)
                {   //    v//n
                    std::string::size_type loc = vertex_indice.find(std::string("//"));
                    std::stringstream transform;
                    transform.str("");
                    transform.clear();
                    transform<<vertex_indice.substr(0,loc);
                    if(!(transform>>pos))
                    {
                        std::cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<std::endl;
                        return false;
                    }
					transform.str("");
					transform.clear();
                    transform<<vertex_indice.substr(loc+2);
                    if(!(transform>>nor))
                    {
                        std::cerr<<"invalid vertx nor in this face "<<"line:"<<line_num<<std::endl;
						std::cerr << vertex_indice << std::endl;
						std::cerr << line << std::endl;
                        return false;
                    }
                    Vertex<Scalar> vertex_temple(pos-1);
                    vertex_temple.setNormalIndex(nor-1);
                    face_temple.addVertex(vertex_temple);
                }
                else
                {
                    if(vertex_indice.find(std::string("/")) == std::string::npos)
                    {
                        std::stringstream transform;
                        transform.str("");
                        transform.clear();
                        transform<<vertex_indice;
                        if(!(transform>>pos))
                        {
                            std::cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<std::endl;
                            return false;
                        }
                        Vertex<Scalar> vertex_temple(pos-1);
                        face_temple.addVertex(vertex_temple);
                    }
                    else 
                    {    //  v/t
                        std::string::size_type loc = vertex_indice.find(std::string("/"));
                        std::stringstream transform;
                        transform.str("");
                        transform.clear();
                        transform<<vertex_indice.substr(0,loc);
                        if(!(transform>>pos))
                        {
                            std::cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<std::endl;
                            return false;
                        }
                        std::string::size_type loc2 = vertex_indice.find(std::string("/"), loc + 1);
                        if(loc2 == std::string::npos)
                        {
                            transform.str("");
                            transform.clear();
                            transform<<vertex_indice.substr(loc+1);
                            if(!(transform>>tex))
                            {
                                std::cerr<<"invalid vertx tex in this face "<<"line:"<<line_num<<std::endl;
                                std::cerr<<vertex_indice<<std::endl;
                                return false;
                            }
                            Vertex<Scalar> vertex_temple(pos-1);
                            vertex_temple.setTextureCoordinateIndex(tex-1);
                            face_temple.addVertex(vertex_temple);
                        }
                        else
                        {
                            transform.str("");
                            transform.clear();
                            transform<<vertex_indice.substr(loc+1,loc2-loc-1);
                            if(!(transform>>tex))
                            {
                                std::cerr<<"invalid vertx tex in this face "<<"line:"<<line_num<<std::endl;
                                std::cerr<<vertex_indice<<std::endl;
                                return false;
                            }
                            transform.str("");
                            transform.clear();
                            transform<<vertex_indice.substr(loc2+1);
                            if(!(transform>>nor))
                            {
                                std::cerr<<"invalid vertx nor in this face "<<"line:"<<line_num<<std::endl;
								std::cerr << line << std::endl;
                                return false;
                            }
                            Vertex<Scalar> vertex_temple(pos-1);
                            vertex_temple.setNormalIndex(nor-1);
                            vertex_temple.setTextureCoordinateIndex(tex-1);
                            face_temple.addVertex(vertex_temple);
                        }
                    }
                }
            }// end while vertex_indices
            num_face++;
            num_group_faces ++;
            current_group->addFace(face_temple);
        }
        else if(type_of_line == std::string("#") || type_of_line == std::string("") )
        {}
        else if(type_of_line == std::string("usemtl"))
        {
            if (num_group_faces > 0)
            {
                // usemtl without a "g" statement : must create a new group
                //first ,create unique name
                std::stringstream clone_name;
                clone_name.clear();
                clone_name<<group_source_name;
                clone_name<<'.';
                clone_name<<group_clone_index;
                std::string new_name;
                clone_name>>new_name;
                mesh->addGroup(FaceGroup<Scalar>(new_name));
                current_group=mesh->groupPtr(new_name);
                num_group_faces = 0;
                group_clone_index++;	
            }
            std::string material_name;
            stream>>material_name;
            if((current_material_index = (mesh->materialIndex(material_name))) != -1)
            {
                if(mesh->numGroups() == 0)
                {
                    mesh->addGroup(FaceGroup<Scalar>(std::string("default")));
                    current_group = mesh->groupPtr(std::string("default"));
                }
                current_group->setMaterialIndex(current_material_index);
            }
            else 
            {
                std::cerr<<"material found false "<<"line:"<<line_num<<std::endl;
                return false;
            }
        }
        else if(type_of_line == std::string("mtllib"))
        {
            std::string mtl_name;
            stream>>mtl_name;
            std::string pre_path = FileUtilities::dirName(filename);
            mtl_name=pre_path+ std::string("/") +mtl_name;
            loadMaterials(mtl_name,mesh);
        }
        else 
        {
            // do nothing
        }		
    }//end while
    ifs.close();
    if(mesh->numMaterials() == 0) //no material read from file, add a default one
        mesh->addMaterial(Material<Scalar>::Iron());
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::save(const std::string &filename, const SurfaceMesh<Scalar> *mesh, bool save_mtl)
{
    if(SurfaceMeshIO<Scalar>::checkFileNameAndMesh(filename,std::string(".obj"),mesh) == false)
        return false;
    std::string prefix = FileUtilities::removeFileExtension(filename);
    std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        std::cerr<<"fail to open file when save a mesh to a obj file"<<std::endl;
        return false;
    }
    std::string material_path = prefix + std::string(".mtl");
    if(save_mtl)
    {
        saveMaterials(material_path, mesh);
        fileout<<"mtllib "<<FileUtilities::filenameInPath(prefix)<<".mtl"<<std::endl;
    }
    unsigned int num_vertices=mesh->numVertices(),i;
    for(i=0;i<num_vertices;++i)
    {
        Vector<Scalar,3> example = mesh->vertexPosition(i);
        fileout<<"v "<<example[0]<<' '<<example[1]<<' '<<example[2]<<std::endl;
    }
    unsigned int num_vertex_normal = mesh->numNormals();
    for(i=0;i<num_vertex_normal;++i)
    {
        Vector<Scalar,3> example = mesh->vertexNormal(i);
        fileout<<"vn "<<example[0]<<' '<<example[1]<<' '<<example[2]<<std::endl;
    }
    unsigned int num_tex = mesh->numTextureCoordinates();
    for(i=0;i<num_tex;++i)
    {
        Vector<Scalar,2> example = mesh->vertexTextureCoordinate(i);
        fileout<<"vt "<<example[0]<<' '<<example[1]<<std::endl;
    }
    unsigned int num_group = mesh->numGroups();
    for(i=0;i<num_group;++i)
    {
        const FaceGroup<Scalar> *group_ptr = mesh->groupPtr(i);
        if(save_mtl)
            fileout<<"usemtl "<<mesh->materialPtr(group_ptr->materialIndex())->name()<<std::endl;
        fileout<<"g "<<group_ptr->name()<<std::endl;
        unsigned int num_face = group_ptr->numFaces(),j;
        const Face<Scalar> *face_ptr;
        for(j=0; j<num_face; ++j)
        {
            face_ptr = group_ptr->facePtr(j);
            fileout<<"f ";
            unsigned int num_vertices_inface = face_ptr->numVertices(),k;
            const Vertex<Scalar> *vertex_ptr;
            for(k=0; k<num_vertices_inface; ++k)
            {
                vertex_ptr = face_ptr->vertexPtr(k);
                fileout<<(vertex_ptr->positionIndex() + 1);
                if(vertex_ptr->hasTexture()||vertex_ptr->hasNormal())
                {
                    fileout<<'/';
                    if(vertex_ptr->hasTexture()) fileout<<vertex_ptr->textureCoordinateIndex() + 1;
                }
                if(vertex_ptr->hasNormal()) fileout<<'/'<<vertex_ptr->normalIndex() + 1<<' ';
            }
            fileout<<std::endl;
        }
    }
    fileout.close();
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::loadMaterials(const std::string &filename, SurfaceMesh<Scalar> *mesh)
{
    if(mesh == NULL)
    {
        std::cerr<<"error:invalid mesh pointer."<<std::endl;
        return false;
    }
    unsigned int line_num = 0;
    std::fstream ifs(filename.c_str(), std::ios::in);
    if(!ifs)
    {
        std::cerr<<"can't open this mtl file"<<std::endl;
        return false;		
    }
    const unsigned int maxline = 1024;
    char line[maxline];
    std::stringstream stream;
    unsigned int num_mtl=0;
    Material<Scalar> material_example;
    while(!ifs.eof())
    {		
        ifs.getline(line, maxline);
        line_num++;
        stream.str("");
        stream.clear();
        char type_of_line[maxline];
        stream<<line;
        stream>>type_of_line;
        std::string texture_file_complete;
        switch (type_of_line[0])
        {
        case '#':
            break;

        case 'n':
            if(num_mtl > 0) mesh->addMaterial(material_example);
            material_example = Material<Scalar>::Iron();
            char mtl_name[maxline];
            stream>>mtl_name;
            num_mtl++;
            material_example.setName(std::string(mtl_name));
            break;

        case 'N':
            if (type_of_line[1] == 's')
            {
                Scalar shininess;
                if(!(stream>>shininess))
                {
                    std::cerr<<"error! no data to set shininess "<<"line:"<<line_num<<std::endl;
                    return false;
                }
                shininess *= 128.0 /1000.0;
                material_example.setShininess(shininess);
            }
            else {}
            break;

        case 'K':
            switch (type_of_line[1])
            {
            case 'd':
                Scalar kd1,kd2,kd3;
                if(!(stream>>kd1)) break;
                stream>>kd2;
                if(!(stream>>kd3))
                {
                    std::cerr<<"error less data when read Kd. "<<"line:"<<line_num<<std::endl;
                    return false;
                }
                material_example.setKd(Vector<Scalar,3> (kd1, kd2, kd3));
                break;

            case 's':
                Scalar ks1,ks2,ks3;
                if(!(stream>>ks1)) break;
                stream>>ks2;
                if(!(stream>>ks3))
                {
                    std::cerr<<"error less data when read Ks. "<<"line:"<<line_num<<std::endl;
                    return false;
                }
                material_example.setKs(Vector<Scalar,3> (ks1, ks2, ks3));
                break;

            case 'a':
                Scalar ka1,ka2,ka3;
                if(!(stream>>ka1)) break;
                stream>>ka2;
                if(!(stream>>ka3))
                {
                    std::cerr<<"error less data when read Ka. "<<"line:"<<line_num<<std::endl;
                    return false;
                }
                material_example.setKa(Vector<Scalar,3> (ka1, ka2, ka3));
                break;

            default:
                break;
            }

        case 'm':
            char tex_name[maxline];
            strcpy(tex_name,"");
            stream>>tex_name;
            texture_file_complete.assign(tex_name);
            texture_file_complete=FileUtilities::dirName(filename)+std::string("/")+texture_file_complete;
            if(strlen(tex_name))material_example.setTextureFileName(texture_file_complete);
            break;

        case 'd':
            char next[maxline];
            Scalar alpha;
            stream>>next;
            if(next[0] == '-') stream>>alpha;
            else 
            {
                stream.clear();
                stream<<next;
                stream>>alpha;
            }
            material_example.setAlpha(alpha);
            break;

        default:
            break;
        }
    }
    //attention at least one material must be in mesh
    mesh->addMaterial(material_example);
    ifs.close();
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::saveMaterials(const std::string &filename, const SurfaceMesh<Scalar> *mesh)
{
    std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        std::cerr<<"error:can't open file when save materials."<<std::endl;
        return false;
    }
    if(mesh == NULL)
    {
        std::cerr<<"error:invalid mesh pointer."<<std::endl;
        return false;
    }
    unsigned int num_mtl = mesh->numMaterials();
    unsigned int i;
    Material<Scalar> material_example;
    for(i = 0;i < num_mtl;i++)
    {
        material_example = mesh->material(i);
        fileout<<"newmtl "<<material_example.name()<<std::endl;
        fileout<<"Ka "<<material_example.Ka()[0]<<' '<<material_example.Ka()[1]<<' '<<material_example.Ka()[2]<<std::endl;
        fileout<<"Kd "<<material_example.Kd()[0]<<' '<<material_example.Kd()[1]<<' '<<material_example.Kd()[2]<<std::endl;
        fileout<<"Ks "<<material_example.Ks()[0]<<' '<<material_example.Ks()[1]<<' '<<material_example.Ks()[2]<<std::endl;
        fileout<<"Ns "<<material_example.shininess()*1000.0/128.0<<std::endl;
        fileout<<"d "<<material_example.alpha()<<std::endl;
        if(material_example.hasTexture()) fileout<<"map_Kd "<<FileUtilities::filenameInPath(material_example.textureFileName())<<std::endl;
    }
    fileout.close();
    return true;
}

//explicit instantitiation
template class ObjMeshIO<float>;
template class ObjMeshIO<double>;

} //end of namespace Physika
