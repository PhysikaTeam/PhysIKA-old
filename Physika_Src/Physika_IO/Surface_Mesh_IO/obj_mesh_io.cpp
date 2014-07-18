/*
* @file obj_mesh_io.cpp 
* @brief load and save mesh to an obj file.
* @author Fei Zhu, Liyou Xu
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
#include <cstring>
#include <sstream>
#include <fstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"
#include "Physika_Core/Utilities/File_Utilities/parse_line.h"

using Physika::SurfaceMeshInternal::Face;
using Physika::SurfaceMeshInternal::Group;
using Physika::SurfaceMeshInternal::Material;
using Physika::SurfaceMeshInternal::Vertex;
using std::endl;
using std::cout;
using std::string;
using std::cerr;

namespace Physika{
    
template <typename Scalar>
bool ObjMeshIO<Scalar>::load(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    if(mesh == NULL)
    {
        std::cout<<"invalid mesh point when call function load()"<<std::endl;
        return false;
    }
    string::size_type suffix_idx = filename.find('.');
    string suffix = filename.substr(suffix_idx);
    if(suffix != string(".obj"))
    {
        std::cout<<"this is not a obj file"<<std::endl;
        return false;
    }
    // a indicator points to current group 
    Group<Scalar>* current_group = NULL;
    // num of total faces
    unsigned int num_face = 0;
    // index of the current material in mesh's materials
    unsigned int current_material_index = 0;
    // the line number in the obj file
    unsigned int line_num = 0;
    unsigned int num_group_faces = 0;
    // group name
    string group_source_name;
    unsigned int group_clone_index = 0;
    std::fstream ifs( filename.c_str(),std::ios::in);
    if(!ifs)
    {
        std::cerr<<"couldn't open "<<filename<<"\n";
        return false;
    }
    string  line;
    std::stringstream stream;
    while(!ifs.eof())
    {
        ++line_num;
        std::getline(ifs,line);
        line = FileUtilities::removeWhitespaces(line);
        string line_next;
        unsigned int line_length=line.size();
        while (line_length > 0 && line[line_length-1] == '\\')     //if the last character in a line is '\',we will merge nextline into this one
        {
            std::getline(ifs,line_next);
            line_next = FileUtilities::removeWhitespaces(line_next);
            line[line_length-1] = ' ';
            line = line + line_next;
            line_length = line.size();
        }
        stream.str("");
        stream.clear();
        stream<<line;
        string type_of_line;
        stream>>type_of_line;
        if(type_of_line[0] == 'v'&& type_of_line.size() == 1)
        {
            //vertex
            Scalar x,y,z;
            if(!(stream>>x))
            {
                cerr<<"error:stream>>x "<<"line:"<<line_num<<endl;
                return false;
            }
            if(!(stream>>y))
            {
                cerr<<"error:stream>>y "<<"line:"<<line_num<<endl;
                return false;
            }
            if(!(stream>>z))
            {
                cerr<<"error:stream>>z "<<"line:"<<line_num<<endl;
                return false;
            }
            mesh->addVertexPosition(Vector<Scalar,3>(x,y,z));
        }
        else if(type_of_line == string("vn"))
        {   //vertex normal
            Scalar x,y,z;
            if(!(stream>>x))
            {
                cerr<<"x position of a normal read error "<<"line:"<<line_num<<endl;
                return false;
            }
            if(!(stream>>y))
            {
                cerr<<"y position of a normal read error "<<"line:"<<line_num<<endl;
                return false;
            }
            if(!(stream>>z))
            {
                cerr<<"z position of a normal read error "<<"line:"<<line_num<<endl;
                return false;
            }
            mesh->addVertexNormal(Vector<Scalar,3>(x,y,z));
        }
        else if(type_of_line == string("vt"))
        {   //vertex texture
            Scalar x,y;
            if(!(stream>>x))
            {
                cerr<<"x position of a texture read error "<<"line:"<<line_num<<endl;
                return false;
            }
            if(!(stream>>y))
            {
                cerr<<"y position of a texture read error "<<"line:"<<line_num<<endl;
                return false;
            }
            mesh->addVertexTextureCoordinate(Vector<Scalar,2>(x,y));
        }
        else if(type_of_line == string("g"))
        {
            string group_name;
            stream>>group_name;
            unsigned int length=group_name.size();
            if(length < 1)
            {
                cout<<"warning: empty group name come in and we consider it as a default group "<<"line:"<<line_num<<endl;
                Group<Scalar> *p=mesh->groupPtr(string("default"));
                if(p == NULL)
                {
                    mesh->addGroup(Group<Scalar>(string("default"),current_material_index));
                    current_group = mesh->groupPtr(string("default"));
                    group_source_name = string("");
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
                mesh->addGroup(Group<Scalar>(group_name,current_material_index));
                current_group=mesh->groupPtr(group_name);
                group_source_name=group_name;
                group_clone_index=0;
                num_group_faces=0;
            }

        }
        else if(type_of_line == string("f") || type_of_line == string("fo"))
        {
            //cout<<line<<endl;
            if(current_group==NULL)
            {
                mesh->addGroup(Group<Scalar>(string("default"),current_material_index));
                current_group=mesh->groupPtr(string("default"));
            }
            Face<Scalar> face_temple;
            string vertex_indice;
            while(stream>>vertex_indice)
            {
                unsigned int pos;
                unsigned int nor;
                unsigned int tex;
                if(vertex_indice.find(string("//")) != string::npos)
                {   //    v//n
                    unsigned int loc = vertex_indice.find(string("//"));
                    std::stringstream transform;
                    transform.str("");
                    transform.clear();
                    transform<<vertex_indice.substr(0,loc);
                    if(!(transform>>pos))
                    {
                        cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<endl;
                        return false;
                    }
                    transform<<vertex_indice.substr(loc+2);
                    if(!(transform>>nor))
                    {
                        cerr<<"invalid vertx nor in this face "<<"line:"<<line_num<<endl;
                        return false;
                    }
                    Vertex<Scalar> vertex_temple(pos-1);
                    vertex_temple.setNormalIndex(nor-1);
                    face_temple.addVertex(vertex_temple);
                }
                else
                {
                    if(vertex_indice.find(string("/")) == string::npos)
                    {
                        std::stringstream transform;
                        transform.str("");
                        transform.clear();
                        transform<<vertex_indice;
                        if(!(transform>>pos))
                        {
                            cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<endl;
                            return false;
                        }
                        Vertex<Scalar> vertex_temple(pos-1);
                        face_temple.addVertex(vertex_temple);
                    }
                    else 
                    {    //  v/t
                        unsigned int loc = vertex_indice.find(string("/"));
                        std::stringstream transform;
                        transform.str("");
                        transform.clear();
                        transform<<vertex_indice.substr(0,loc);
                        if(!(transform>>pos))
                        {
                            cerr<<"invalid vertx pos in this face "<<"line:"<<line_num<<endl;
                            return false;
                        }
                        unsigned int loc2 = vertex_indice.find(string("/"), loc + 1);
                        if(loc2 == string::npos)
                        {
                            transform.str("");
                            transform.clear();
                            transform<<vertex_indice.substr(loc+1);
                            if(!(transform>>tex))
                            {
                                cerr<<"invalid vertx tex in this face "<<"line:"<<line_num<<endl;
                                cerr<<vertex_indice<<endl;
                                cerr<<"haha"<<endl;
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
                                cerr<<"invalid vertx tex in this face "<<"line:"<<line_num<<endl;
                                cerr<<vertex_indice<<endl;
                                return false;
                            }
                            transform.str("");
                            transform.clear();
                            transform<<vertex_indice.substr(loc2+1);
                            if(!(transform>>nor))
                            {
                                cerr<<"invalid vertx nor in this face "<<"line:"<<line_num<<endl;
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
        else if(type_of_line == string("#") || type_of_line == string("") ){}
        else if(type_of_line == string("usemtl"))
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
                string new_name;
                clone_name>>new_name;
                mesh->addGroup(Group<Scalar>(new_name));
                current_group=mesh->groupPtr(new_name);
                num_group_faces = 0;
                group_clone_index++;	
            }
            string material_name;
            stream>>material_name;
            if((current_material_index = (mesh->materialIndex(material_name))) != -1)
            {
                if(mesh->numGroups() == 0)
                {
                    mesh->addGroup(Group<Scalar>(string("default")));
                    current_group = mesh->groupPtr(string("default"));
                }
                current_group->setMaterialIndex(current_material_index);
            }
            else 
            {
                cerr<<"material found false "<<"line:"<<line_num<<endl;
                return false;
            }
        }
        else if(type_of_line == string("mtllib"))
        {
            string mtl_name;
            stream>>mtl_name;
            string pre_path = FileUtilities::dirName(filename);
            mtl_name=pre_path+ string("/") +mtl_name;
            loadMaterials(mtl_name,mesh);
        }
        else 
        {
            // do nothing
        }		
    }//end while
    ifs.close();
    //cout some file message
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::save(const string &filename, const SurfaceMesh<Scalar> *mesh)
{
    if(mesh == NULL)
    {
        cerr<<"invalid mesh point"<<endl;
        return false;
    }
    string::size_type suffix_idx = filename.find('.');
    string suffix = filename.substr(suffix_idx), prefix = filename.substr(0, suffix_idx);
    if(suffix != string(".obj"))
    {
        cerr<<"this is not a obj file"<<endl;
        return false;
    }	
    std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        cerr<<"fail to open file when save a mesh to a obj file"<<endl;
        return false;
    }
    string material_path = prefix + string(".mtl");
    saveMaterials(material_path, mesh);
    fileout<<"mtllib "<<FileUtilities::filenameInPath(prefix)<<".mtl"<<endl;
    unsigned int num_vertices=mesh->numVertices(),i;
    for(i=0;i<num_vertices;++i)
    {
        Vector<Scalar,3> example = mesh->vertexPosition(i);
        fileout<<"v "<<example[0]<<' '<<example[1]<<' '<<example[2]<<endl;
    }
    unsigned int num_vertex_normal = mesh->numNormals();
    for(i=0;i<num_vertex_normal;++i)
    {
        Vector<Scalar,3> example = mesh->vertexNormal(i);
        fileout<<"vn "<<example[0]<<' '<<example[1]<<' '<<example[2]<<endl;
    }
    unsigned int num_tex = mesh->numTextureCoordinates();
    for(i=0;i<num_tex;++i)
    {
        Vector<Scalar,2> example = mesh->vertexTextureCoordinate(i);
        fileout<<"vt "<<example[0]<<' '<<example[1]<<endl;
    }
    unsigned int num_group = mesh->numGroups();
    for(i=0;i<num_group;++i)
    {
        const Group<Scalar> *group_ptr = mesh->groupPtr(i);
        fileout<<"usemtl "<<mesh->materialPtr(group_ptr->materialIndex())->name()<<endl;
        fileout<<"g "<<group_ptr->name()<<endl;
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
            fileout<<endl;
        }
    }
    fileout.close();
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::loadMaterials(const string &filename, SurfaceMesh<Scalar> *mesh)
{
    if(mesh == NULL)
    {
        cerr<<"error:invalid mesh point."<<endl;
        return false;
    }
    unsigned int line_num = 0;
    std::fstream ifs(filename.c_str(), std::ios::in);
    if(!ifs)
    {
        cerr<<"can't open this mtl file"<<endl;
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
        string texture_file_complete;
        switch (type_of_line[0])
        {
        case '#':
            break;

        case 'n':
            if(num_mtl > 0) mesh->addMaterial(material_example);
            material_example.setKa(Vector<Scalar,3> (0.1, 0.1, 0.1));
            material_example.setKd(Vector<Scalar,3> (0.5, 0.5, 0.5));
            material_example.setKs(Vector<Scalar,3> (0.0, 0.0, 0.0));
            material_example.setShininess(65);
            material_example.setAlpha(1);
            material_example.setTextureFileName(string());
            char mtl_name[maxline];
            stream>>mtl_name;
            num_mtl++;
            material_example.setName(string(mtl_name));
            break;

        case 'N':
            if (type_of_line[1] == 's')
            {
                Scalar shininess;
                if(!(stream>>shininess))
                {
                    cerr<<"error! no data to set shininess "<<"line:"<<line_num<<endl;
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
                    cerr<<"error less data when read Kd. "<<"line:"<<line_num<<endl;
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
                    cerr<<"error less data when read Ks. "<<"line:"<<line_num<<endl;
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
                    cerr<<"error less data when read Ka. "<<"line:"<<line_num<<endl;
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
            texture_file_complete=FileUtilities::dirName(filename)+string("/")+texture_file_complete;
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
    if(num_mtl >= 0)                            //attention at least one material must be in mesh
        mesh->addMaterial(material_example);
    ifs.close();
    return true;
}

template <typename Scalar>
bool ObjMeshIO<Scalar>::saveMaterials(const string &filename, const SurfaceMesh<Scalar> *mesh)
{
    std::fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        cerr<<"error:can't open file when save materials."<<endl;
        return false;
    }
    if(mesh == NULL)
    {
        cerr<<"error:invalid mesh point."<<endl;
        return false;
    }
    unsigned int num_mtl = mesh->numMaterials();
    unsigned int i;
    Material<Scalar> material_example;
    for(i = 0;i < num_mtl;i++)
    {
        material_example = mesh->material(i);
        fileout<<"newmtl "<<material_example.name()<<endl;
        fileout<<"Ka "<<material_example.Ka()[0]<<' '<<material_example.Ka()[1]<<' '<<material_example.Ka()[2]<<endl;
        fileout<<"Kd "<<material_example.Kd()[0]<<' '<<material_example.Kd()[1]<<' '<<material_example.Kd()[2]<<endl;
        fileout<<"Ks "<<material_example.Ks()[0]<<' '<<material_example.Ks()[1]<<' '<<material_example.Ks()[2]<<endl;
        fileout<<"Ns "<<material_example.shininess()*1000.0/128.0<<endl;
        fileout<<"d "<<material_example.alpha()<<endl;
        if(material_example.hasTexture()) fileout<<"map_Kd "<<FileUtilities::filenameInPath(material_example.textureFileName())<<endl;
    }
    fileout.close();
    return true;
}

//explicit instantitation
template class ObjMeshIO<float>;
template class ObjMeshIO<double>;

} //end of namespace Physika
