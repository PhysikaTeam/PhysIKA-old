/*
 * @file volumetric_mesh_io.cpp 
 * @brief volumetric mesh loader/saver, load/save volumetric mesh from/to file.
 * @author Fei Zhu, liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Core/Utilities/File_Utilities/file_content_utilities.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"

using std::string;
using std::fstream;
using std::getline;
using std::vector;

namespace Physika{

using VolumetricMeshIOInternal::SaveOption;
using VolumetricMeshIOInternal::SINGLE_FILE;
using VolumetricMeshIOInternal::SEPARATE_FILES;
using VolumetricMeshIOInternal::ZERO_INDEX;
using VolumetricMeshIOInternal::ONE_INDEX;

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>* VolumetricMeshIO<Scalar,Dim>::load(const string &filename)
{
    VolumetricMesh<Scalar, Dim> *pointer = NULL;           //pointer we will return 
    string dir = FileUtilities::dirName(filename);
    string file_extension = FileUtilities::fileExtension(filename);
    if(file_extension.size() == 0)
    {
        std::cerr<<"No file extension found for the mesh file:"<<filename<<std::endl;
        return NULL;
    }
    if(file_extension != string(".smesh"))
    {
        std::cerr<<"Unknown mesh file format:"<<file_extension<<std::endl;
        return NULL;
    }
    vector<fstream *> file_stack;
    fstream *fp = new fstream;
    fp->open(filename.c_str(),std::ios::in);
    if(!(*fp))
    {
        std::cerr<<"Couldn't opern .smesh file:"<<filename<<std::endl;
        return NULL;
    }
    // first check the mesh type
    string mesh_type;
    while(*fp >>mesh_type)
        if(mesh_type == string("*ELEMENTS"))break;
    *fp >>mesh_type;
    fp->seekg(std::ios::beg);
    if(mesh_type == string("TET"))
    {
        pointer = dynamic_cast<VolumetricMesh<Scalar, Dim>*> (new TetMesh<Scalar>());
    }
    else if(mesh_type == string("CUBIC"))
    {
        pointer = dynamic_cast<VolumetricMesh<Scalar, Dim>*> (new CubicMesh<Scalar>());
    }
    else if(mesh_type == string("TRI"))
    {
        pointer = dynamic_cast<VolumetricMesh<Scalar, Dim>*> (new TriMesh<Scalar>());
    }
    else if(mesh_type == string("QUAD"))
    {
        pointer = dynamic_cast<VolumetricMesh<Scalar, Dim>*> (new QuadMesh<Scalar>());
    }
    else if(mesh_type == string("NONUNIFORM"))
    {
        //TO DO: LOAD NONUNIFORM VOLUMETRIC MESHES
        std::cerr<<"Non-uniform element type not implemented yet.\n";
        return NULL;
    }
    else 
    {
        std::cerr<<"Unknow elements type:"<<mesh_type<<std::endl;
        return NULL;
    }
    enum ParseSession{
        NOT_SET,
        VERTICES,
        ELEMENTS,
        REGIONS
    };
    ParseSession cur_session = NOT_SET; //current state of parse
    int index_start_vert = 0, index_start_ele = 0; // 0 or 1   .smesh file is 0_indexed or 1_indexed?
    string line_str;
    unsigned int vert_num = 0;
    int vert_dim = 0;
    unsigned int ele_num = 0;
    int ele_dim = 0;
    std::vector<unsigned int> region;
    bool firstline_aftercommand = false;
    bool mesh_kind_decided = false;
    bool start_index_decided = false;
    string mesh_kind;
    string region_name;
    std::stringstream str_in;
    while((!fp->eof())||(!file_stack.empty()))
    {
        if(fp->eof())  //already reach the file end,we should return the last file.
        {
            fp->close();
            delete fp;
            fp = file_stack.back();
            file_stack.pop_back();
            continue;               //continue to read nextline
        }
        getline(*fp,line_str);
        //first remove preceding blanks from this line
        line_str = FileUtilities::removeWhitespaces(line_str,1);
        str_in.clear();
        str_in.str("");
        str_in<<line_str;
        if(line_str.empty())  //empty line
            continue;
        if(line_str[0] == '#')  //comment
            continue;
        if(line_str[0] == '*')  //commands
        {
            if(line_str.substr(0,9) == string("*VERTICES")) //vertices
            {
                if(!region.empty())
                {
                    pointer->addRegion(region_name,region);
                    region.clear();
                }
                cur_session = VERTICES;
                firstline_aftercommand = true;
            }
            else if(line_str.substr(0,9) == string("*ELEMENTS")) //elements
            {
                if(!region.empty())
                {
                    pointer->addRegion(region_name,region);
                    region.clear();
                }
                cur_session = ELEMENTS;
                mesh_kind_decided = true;
            }
            else if(line_str.substr(0,7) == string("*REGION")) //set
            {
                if(!region.empty())
                {
                    pointer->addRegion(region_name,region);
                    region.clear();
                }
                cur_session = REGIONS;
                region_name = line_str.substr(8);
                region.clear();
                //first_region = false;
            }
            else if(line_str.substr(0,8) == string("*INCLUDE")) //include
            {
                string complete_file_name = dir + string("/") + line_str.substr(9);

                std::cout<<complete_file_name<<std::endl;           //check
                file_stack.push_back(fp);
                fp = new fstream;
                fp->open(complete_file_name.c_str(),std::ios::in);
                if(!*fp)
                {
                    std::cerr<<"fail to open file:"<<complete_file_name<<std::endl;
                    return NULL;
                }
                continue;
            }
            else
            {
                std::cerr<<"illeagal command:"<<line_str<<std::endl;
                return NULL;
            }
        }
        else  //
        {
            if(cur_session == VERTICES)
            {
                if(firstline_aftercommand)
                {
                    str_in>>vert_num;
                    str_in>>vert_dim;
                    if(vert_dim != Dim)
                    {
                        std::cerr<<"dimension unmatched!"<<std::endl;
                        return NULL;
                    }
                    //vertices = new Scalar[vert_num*vert_dim];
                    start_index_decided = true;
                    firstline_aftercommand = false;
                }
                else
                {
                    int vert_index;
                    Vector<Scalar, Dim> vertex_;
                    if(start_index_decided)
                    {
                        str_in>>index_start_vert;
                        for(int i=0; i<vert_dim; ++i) str_in>>vertex_[i];
                        pointer->addVertex(vertex_);
                        start_index_decided = false;
                    }
                    else
                    {
                        str_in>>vert_index;
                        for(int i=0; i<vert_dim; ++i) str_in>>vertex_[i];
                        pointer->addVertex(vertex_);
                    }
                }
            }
            else if(cur_session == ELEMENTS)
            {
                if(mesh_kind_decided)
                {
                    str_in>>mesh_kind;
                    firstline_aftercommand = true;
                    mesh_kind_decided = false;
                }
                else
                {
                    if(firstline_aftercommand)
                    {
                        str_in>>ele_num;
                        str_in>>ele_dim;
                        //elements = new unsigned int[ele_num * ele_dim];
                        firstline_aftercommand = false;
                        start_index_decided =true;
                    }
                    else
                    {
                        int ele_index;
                        unsigned int ele;
                        vector<unsigned int> elements;
                        elements.clear();
                        if(start_index_decided)
                        {
                            str_in>>index_start_ele;
                            for(int i=0; i<ele_dim; i++)
                            {
                                str_in>> ele;
                                elements.push_back(ele - index_start_vert);
                            }
                            pointer->addElement(elements);
                            start_index_decided = false;
                        }
                        else
                        {
                            str_in>>ele_index;
                            for(int i=0; i<ele_dim; ++i)
                            {
                                str_in>>ele;
                                elements.push_back(ele - index_start_ele);
                            }
                            pointer->addElement(elements);
                        }
                    }
                }
            }
            else if(cur_session == REGIONS)
            {
                unsigned int region_index;
                char comma;
                if(line_str[0]==',')str_in>>comma;
                while(str_in>>region_index)
                {
                    region.push_back(region_index - index_start_ele);
                    str_in>>comma;
                }
            }
        }
    }
    if(!region.empty())
    {
        pointer->addRegion(region_name,region);
        region.clear();
    }
    if(pointer->regionNum() == 0) //no region read from file, add one region named "AllElements"
    {
        region.resize(pointer->eleNum());
        for(unsigned int ele_idx = 0; ele_idx < pointer->eleNum(); ++ele_idx)
            region[ele_idx] = ele_idx;
        region_name = string("AllElements");
        pointer->addRegion(region_name,region);
    }
    return pointer;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::save(const string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh)
{
    return save(filename,volumetric_mesh,SINGLE_FILE|ONE_INDEX);
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::save(const string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, SaveOption option)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension specified for the mesh file!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix!=string(".smesh"))
    {
        std::cerr<<"Unknown mesh file format specified!\n";
        return false;
    }
    if(volumetric_mesh == NULL)
    {
        std::cerr<<"VolumetricMesh pointer is NULL!\n";
        return false;
    }
    resolveInvalidOption(option);
    unsigned int start_index = 1;
    if(option & ZERO_INDEX)
        start_index = 0;
    else if(option & ONE_INDEX)
        start_index = 1;
    else
        PHYSIKA_ERROR("Invalid option!");
    if(option & SINGLE_FILE)
        return saveToSingleFile(filename,volumetric_mesh,start_index);
    else if(option & SEPARATE_FILES)
        return saveToSeparateFiles(filename,volumetric_mesh,start_index);
    else
        PHYSIKA_ERROR("Invalid option!");
    return false;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::saveToSingleFile(const string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, unsigned int start_index)
{
    PHYSIKA_ASSERT(volumetric_mesh);
    PHYSIKA_ASSERT(start_index==0||start_index==1);
    fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        std::cerr<<"Failed to create file "<<filename<<"!\n";
        return false;
    }
    fileout<<"# "<<filename<<"\n";
    fileout<<"# Generated by Physika\n";
    //vertices
    fileout<<"*VERTICES\n";
    fileout<<volumetric_mesh->vertNum()<<" "<<Dim<<" 0 0\n";
    for(unsigned int i = start_index; i < volumetric_mesh->vertNum()+start_index; ++i)
    {
        Vector<Scalar,Dim> vert_pos = volumetric_mesh->vertPos(i-start_index);
        fileout<<i<<" ";
        for(int j = 0; j < Dim; ++j)
            fileout<<vert_pos[j]<<" ";
        fileout<<"\n";
    }
    //elements
    fileout<<"*ELEMENTS\n";
    switch(volumetric_mesh->elementType())
    {
    case VolumetricMeshInternal::TRI:
        fileout<<"TRI\n";
        break;
    case VolumetricMeshInternal::QUAD:
        fileout<<"QUAD\n";
        break;
    case VolumetricMeshInternal::TET:
        fileout<<"TET\n";
        break;
    case VolumetricMeshInternal::CUBIC:
        fileout<<"CUBIC\n";
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        fileout<<"NONUNIFORM\n";
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
    fileout<<volumetric_mesh->eleNum()<<" ";
    if(volumetric_mesh->isUniformElementType())
        fileout<<volumetric_mesh->eleVertNum()<<" 0\n";
    else
        fileout<<"-1 0\n";
    for(unsigned int i = start_index; i < volumetric_mesh->eleNum()+start_index; ++i)
    {
        fileout<<i<<" ";
        for(unsigned int j = 0; j < volumetric_mesh->eleVertNum(i-start_index); ++j)
            fileout<<volumetric_mesh->eleVertIndex(i-start_index,j)+start_index<<" ";
        fileout<<"\n";
    }
    //regions
    unsigned int region_num = volumetric_mesh->regionNum();
    if(region_num>1)
    {
        for(unsigned int i = 0; i < region_num; ++i)
        {
            fileout<<"*REGION "<<volumetric_mesh->regionName(i)<<"\n";
            vector<unsigned int> region_elements;
            volumetric_mesh->regionElements(i,region_elements);
            for(unsigned int j = 0; j < region_elements.size(); ++j)
            {
                fileout<<region_elements[j]+start_index;
                if(j==region_elements.size()-1)
                    fileout<<"\n";
                else
                    fileout<<", ";
            }
        }
    }
    fileout.close();
    return true;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::saveToSeparateFiles(const string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, unsigned int start_index)
{
    PHYSIKA_ASSERT(volumetric_mesh);
    PHYSIKA_ASSERT(start_index==0||start_index==1);
    int name_base_length = filename.length() - 6;
    string name_base = filename.substr(0,name_base_length);
    string node_filename = name_base + string(".node");
    string ele_filename = name_base + string(".ele");
    ////////////////////////////////////////.smesh file///////////////////////////////////////////
    fstream smesh_fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!smesh_fileout)
    {
        std::cerr<<"Failed to create file "<<filename<<"!\n";
        return false;
    }
    smesh_fileout<<"# "<<filename<<"\n";
    smesh_fileout<<"# Generated by Physika\n";
    //vertices
    smesh_fileout<<"*VERTICES\n";
    smesh_fileout<<"*INCLUDE "<<node_filename<<"\n";
    //elements
    smesh_fileout<<"*ELEMENTS\n";
    switch(volumetric_mesh->elementType())
    {
    case VolumetricMeshInternal::TRI:
        smesh_fileout<<"TRI\n";
        break;
    case VolumetricMeshInternal::QUAD:
        smesh_fileout<<"QUAD\n";
        break;
    case VolumetricMeshInternal::TET:
        smesh_fileout<<"TET\n";
        break;
    case VolumetricMeshInternal::CUBIC:
        smesh_fileout<<"CUBIC\n";
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        smesh_fileout<<"NONUNIFORM\n";
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
    smesh_fileout<<"*INCLUDE "<<ele_filename<<"\n";
    //regions
    unsigned int region_num = volumetric_mesh->regionNum();
    if(region_num>1)
    {
        for(unsigned int i = 0; i < region_num; ++i)
        {
            smesh_fileout<<"*REGION "<<volumetric_mesh->regionName(i)<<"\n";
            vector<unsigned int> region_elements;
            volumetric_mesh->regionElements(i,region_elements);
            for(unsigned int j = 0; j < region_elements.size(); ++j)
            {
                smesh_fileout<<region_elements[j]+start_index;
                if(j==region_elements.size()-1)
                    smesh_fileout<<"\n";
                else
                    smesh_fileout<<", ";
            }
        }
    }
    smesh_fileout.close();
    ////////////////////////////////////////.node file///////////////////////////////////////////
    fstream node_fileout(node_filename.c_str(),std::ios::out|std::ios::trunc);
    if(!node_fileout)
    {
        std::cerr<<"Failed to create file "<<node_filename<<"!\n";
        return false;
    }
    node_fileout<<volumetric_mesh->vertNum()<<" "<<Dim<<" 0 0\n";
    for(unsigned int i = start_index; i < volumetric_mesh->vertNum()+start_index; ++i)
    {
        Vector<Scalar,Dim> vert_pos = volumetric_mesh->vertPos(i-start_index);
        node_fileout<<i<<" ";
        for(int j = 0; j < Dim; ++j)
            node_fileout<<vert_pos[j]<<" ";
        node_fileout<<"\n";
    }
    node_fileout.close();
    ////////////////////////////////////////.ele file///////////////////////////////////////////
    fstream ele_fileout(ele_filename.c_str(),std::ios::out|std::ios::trunc);
    if(!ele_fileout)
    {
        std::cerr<<"Failed to create file "<<ele_filename<<"!\n";
        return false;
    }
    ele_fileout<<volumetric_mesh->eleNum()<<" ";
    if(volumetric_mesh->isUniformElementType())
        ele_fileout<<volumetric_mesh->eleVertNum()<<" 0\n";
    else
        ele_fileout<<"-1 0\n";
    for(unsigned int i = start_index; i < volumetric_mesh->eleNum()+start_index; ++i)
    {
        ele_fileout<<i<<" ";
        for(unsigned int j = 0; j < volumetric_mesh->eleVertNum(i-start_index); ++j)
            ele_fileout<<volumetric_mesh->eleVertIndex(i-start_index,j)+start_index<<" ";
        ele_fileout<<"\n";
    }
    ele_fileout.close();
    return true;
}

template <typename Scalar, int Dim>
void VolumetricMeshIO<Scalar,Dim>::resolveInvalidOption(SaveOption &option)
{
    if((option & SINGLE_FILE) && (option & SEPARATE_FILES))
    {
        std::cerr<<"Conflict options of SINGLE_FILE and SEPARATE_FILES, use SINGLE_FILE.\n";
        option &= ~SEPARATE_FILES;
    }
    if((option & ZERO_INDEX) && (option & ONE_INDEX))
    {
        std::cerr<<"Conflict options of ZERO_INDEX and ONE_INDEX, use ONE_INDEX.\n";
        option &= ~ZERO_INDEX;
    }
    if(((option & SINGLE_FILE) || (option & SEPARATE_FILES)) == false)
    {
        std::cerr<<"Neither SINGLE_FILE nor SEPARATE_FILES is set, use SINGLE_FILE.\n";
        option |= SINGLE_FILE;
    }
    if(((option & ZERO_INDEX) || (option & ONE_INDEX)) == false)
    {
        std::cerr<<"Neither ZERO_INDEX nor ONE_INDEX is set, use ONE_INDEX.\n";
        option |= ONE_INDEX;
    }
}

//explicit instantiations
template class VolumetricMeshIO<float,2>;
template class VolumetricMeshIO<float,3>;
template class VolumetricMeshIO<double,2>;
template class VolumetricMeshIO<double,3>;

}  //end of namespace Physika
