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
	vector<unsigned int> regionEleIdx;
	TriMesh<float> obj(3,a,1,element);
    //VolumetricMeshIO<float,2>::save("tri_mesh.smesh",&obj);
    //VolumetricMeshIO<float,2>::save("tri_mesh.smesh",&obj,SEPARATE_FILES|ZERO_INDEX);

	Physika::VolumetricMesh<float, 3> *p;
	p = VolumetricMeshIO<float, 3>::load(string("C:/Users/acer/Documents/Tencent Files/731595774/FileRecv/bar.smesh"));
	p->printInfo();
	cout<<"vertNum:"<<p->vertNum()<<endl;
	cout<<"eleNum:"<<p->eleNum()<<endl;
	cout<<"regionNum:"<<p->regionNum()<<endl;
	cout<<p->eleVertIndex(0,0)<<endl;
	cout<<p->regionName(0)<<endl;
	p->regionElements(p->regionName(0),regionEleIdx);
	for(int i=0;i<regionEleIdx.size();++i)cout<<regionEleIdx[i]<<" ";
	cout<<endl;
	getchar();
    return 0;
}
