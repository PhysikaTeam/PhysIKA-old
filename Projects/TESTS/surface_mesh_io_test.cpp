/*
 * @file surface_mesh_io.cpp 
 * @brief test  surface mesh io.
 * @author LiYou Xu, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2014 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include<iostream>
#include<string>
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
using Physika::SurfaceMeshIO;
using Physika::ObjMeshIO;
using Physika::SurfaceMesh;
using namespace std;


int main()
{
    string input_filename,output_filename;
    SurfaceMesh<float> mesh;
    cout<<"Input the file name to load: \n";
    cin>>input_filename;
    cout<<"Loading mesh from "<<input_filename<<"\n";
    bool status = SurfaceMeshIO<float>::load(input_filename,&mesh);
    if(status)
    {
        cout<<"Success! \n";
        int option = 2;
        cout<<"Display mesh information ? 1: Yes, 2: No\n";
        cin>>option;
        if(option == 1)
        {
            int vertex_num = mesh.numVertices(),normal_num = mesh.numNormals(), texture_num = mesh.numTextureCoordinates();
            cout<<"vertex_num:"<<vertex_num<<endl;
            for(int i = 0;i<vertex_num;++i)
                cout<<mesh.vertexPosition(i)<<endl;
            cout<<"normal_num:"<<normal_num<<endl;
            for(int i=0;i<normal_num;++i)
                cout<<mesh.vertexNormal(i)<<endl;
            cout<<"texture_num:"<<texture_num<<endl;
            for(int i=0;i<texture_num;++i)
                cout<<mesh.vertexTextureCoordinate(i)<<endl;
            int group_num=mesh.numGroups();
            cout<<"group_num:"<<group_num<<endl;
            for(int i=0;i<group_num;++i)
            {
                cout<<i<<' '<<mesh.group(i).name()<<' '<<mesh.group(i).materialIndex()<<endl;
            }
            int mtl_num=mesh.numMaterials();
            for(int i=0;i<mtl_num;++i)
                cout<<i<<"   "<<mesh.material(i).name()<<" "<<mesh.material(i).textureFileName() <<" Ns:"<<mesh.material(i).shininess()<<endl;
        }
        option = 2;
        cout<<"Save mesh file? 1: Yes, 2: No\n";
        cin>>option;
        if(option == 1)
        {
            cout<<"Input the file name to save:\n";
            cin>>output_filename;
            status = SurfaceMeshIO<float>::save(output_filename,&mesh);
            if(status)
                cout<<"Success!\n";
        }
    }
    return 0;
}
