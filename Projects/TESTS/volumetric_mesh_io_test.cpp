/*
 * @file volumetric_mesh_io_test.cpp 
 * @brief Test the IO operations of volumetric meshes.
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
#include <vector>
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
using namespace std;
using Physika::VolumetricMeshIO;
using Physika::TriMesh;
using Physika::VolumetricMeshIOInternal::SEPARATE_FILES;
using Physika::VolumetricMeshIOInternal::ZERO_INDEX;

int main()
{
	float a[6]={0,0,0,1,1,0};
	unsigned int element[3]={0,1,2};
	int b = 3;
	TriMesh<float> obj(3,a,1,element);
    //VolumetricMeshIO<float,2>::save("tri_mesh.smesh",&obj);
    //VolumetricMeshIO<float,2>::save("tri_mesh.smesh",&obj,SEPARATE_FILES|ZERO_INDEX);

	Physika::VolumetricMesh<float, 2> *p;
	p = VolumetricMeshIO<float, 2>::load(string("tri_mesh.smesh"));
	p->printInfo();
	cout<<"vertNum:"<<p->vertNum()<<endl;
	cout<<"eleNum:"<<p->eleNum()<<endl;
	cout<<"regionNum:"<<p->regionNum()<<endl;
	cout<<"region:"<<p->regionName(0)<<endl;
	vector<unsigned int> v ;
	p->regionElements(string("first"),v );
	for(int i=0;i<v.size();++i)cout<<v[i]<<endl;
	getchar();
    return 0;
}
