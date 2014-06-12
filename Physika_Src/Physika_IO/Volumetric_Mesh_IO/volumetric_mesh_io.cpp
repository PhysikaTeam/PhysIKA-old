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
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
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
    return false;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::saveToSingleFile(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh)
{
    return false;
}

template <typename Scalar, int Dim>
bool VolumetricMeshIO<Scalar,Dim>::saveToSeparateFiles(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh)
{
    return false;
}
//explicit instantiations
template class VolumetricMeshIO<float,2>;
template class VolumetricMeshIO<float,3>;
template class VolumetricMeshIO<double,2>;
template class VolumetricMeshIO<double,3>;

}  //end of namespace Physika
