/*
 * @file obj_surface_mesh_io.cpp 
 * @brief Test functions in obj_surface_mesh_io.cpp whether can successfully read and save an obj file.
 * @author LiYou Xu
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
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
using Physika::ObjMeshIO;
using Physika::SurfaceMesh;
using std::cout;
using std::endl;
int main()
{
	SurfaceMesh<float> mesh;
	ObjMeshIO<float>::load(string("C:/Users/acer/Documents/model/baolingqiu.obj"), &mesh);
	ObjMeshIO<float>::save(string("C:/Users/acer/Documents/model/baolingqiu_fuben.obj"),&mesh);
	/*
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
	*/
	getchar();
	return 0;
}