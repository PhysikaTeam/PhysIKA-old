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

#include <iostream>
#include <fstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"

using std::string;
using std::fstream;
using std::getline;

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>* VolumetricMeshIO<Scalar,Dim>::load(const string &filename)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx >= filename.size())
    {
        std::cerr<<"No file extension found for the mesh file!\n";
        return NULL;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix != string(".smesh"))
    {
        std::cerr<<"Unknown mesh file format!\n";
        return NULL;
    }
    fstream ifs(filename.c_str(),std::ios::in);
    if(!ifs)
    {
        std::cerr<<"Couldn't opern .smesh file!\n";
        return NULL;
    }
    enum ParseSession{
        NOT_SET,
        VERTICES,
        ELEMENTS,
        SETS
    };
    ParseSession cur_session = NOT_SET; //current state of parse
    int index_start = 0; // 0 or 1
    string line_str;
    int vert_num = 0;
    Scalar *vertices = NULL;
    int ele_num = 0;
    int *elements = NULL;
    while(getline(ifs,line_str))
    {
        //first remove preceding blanks from this line
        string::size_type first_no_blank_idx = line_str.find(' ');
        line_str = line_str.substr(first_no_blank_idx);
        if(line_str.empty())  //empty line
            continue;
        if(line_str[0] == '#')  //comment
            continue;
        if(line_str[0] == '*')  //commands
        {
            if(line_str.substr(0,9) == string("*VERTICES")) //vertices
            {
                cur_session = VERTICES;
            }
            else if(line_str.substr(0,9) == string("*ELEMENTS")) //elements
            {
                cur_session = ELEMENTS;
            }
            else if(line_str.substr(0,4) == string("*SET")) //set
            {
                cur_session = SETS;
            }
            else if(line_str.substr(0,8) == string("*INCLUDE")) //include
            {
            }
            else
            {
            }
        }
        else  //
        {
        }
    }
    return NULL;
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
bool VolumetricMeshIO<Scalar,Dim>::saveToSingleFile(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, unsigned int start_index)
{
    fstream fileout(filename.c_str(),std::ios::out|std::ios::trunc);
    if(!fileout)
    {
        std::cerr<<"Failed to create file "<<filename<<"!\n";
        return false;
    }
    PHYSIKA_ASSERT(volumetric_mesh);
    PHYSIKA_ASSERT(start_index==0||start_index==1);
    fileout<<"# "<<filename<<"\n";
    fileout<<"# Generated by Physika\n";
    fileout<<"*VERTICES\n";
    fileout<<volumetric_mesh->vertNum()<<" "<<Dim<<" 0 0\n";
    for(int i = start_index; i < volumetric_mesh->vertNum()+start_index; ++i)
    {
        Vector<Scalar,Dim> vert_pos = volumetric_mesh->vertPos(i-start_index);
        fileout<<i<<" ";
        for(int j = 0; j < Dim; ++j)
            fileout<<vert_pos[j]<<" ";
        fileout<<"\n";
    }
    //TO DO
    fileout.close();
    return false;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::saveToSeparateFiles(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, unsigned int start_index)
{
    return false;
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
    if((option & SINGLE_FILE) || (option & SEPARATE_FILES) == false)
    {
        std::cerr<<"Neither SINGLE_FILE nor SEPARATE_FILES is set, use SINGLE_FILE.\n";
        option |= SINGLE_FILE;
    }
    if((option & ZERO_INDEX) || (option & ONE_INDEX) == false)
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
