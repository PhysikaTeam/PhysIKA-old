/*
 * @file volumetric_mesh_test.cpp 
 * @brief Test the various types of volumetric meshes.
 * @author Liyou Xu, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 * contact me at mikepkucs@gmail.com if any question
 */

#include <iostream>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Core/Vectors/vector_2d.h"
using namespace std;
using Physika::VolumetricMesh;
using Physika::TetMesh;
using Physika::TriMesh;
using Physika::CubicMesh;
using Physika::QuadMesh;


int main()
{
	cout<<"triMesh part:"<<endl;
	float a[6]={0,0,0,1,1,0};
	unsigned int element[3]={0,1,2};
	int b = 3;
	TriMesh<float> obj(3,a,1,element);
	cout<<"ele_num:"<<obj.eleNum()<<endl;
	cout<<"is uniform:"<<obj.isUniformElementType()<<endl;
	cout<<"3 vertices:"<<endl;
	cout<<obj.eleVertPos(0,0)<<' '<<obj.eleVertPos(0,1)<<' '<<obj.eleVertPos(0,2)<<endl;
	
	cout<<"volume: "<<obj.eleVolume(0) <<endl;
	cout<<"contain point(0.5,0.5): "<<obj.containsVertex(0,Physika::Vector<float, 2>(0.5,0.5) )<<endl;
	float weights[3];
	obj.interpolationWeights( 0,Physika::Vector<float,2>(0.1,0.9),weights);
	cout<<"interplation(0.1,0.9):"<<weights[0]<<' '<<weights[1]<<' '<<weights[2]<<endl;
	cout<<endl;

	cout<<"tetMesh part:"<<endl;
	double tetvp[21]={0,0,0, 0,0,1, 0,1,0, 1,0,0, 0,0,2, 0,2,0, 2,0,0};
	unsigned int eletet[8]={0,1,2,3,0,4,5,6};
	TetMesh<double> tetobj(7,tetvp,2,eletet);
	cout<<"contain point(0.1,0.2,0.3): "<<tetobj.containsVertex(0,Physika::Vector<double,3>(0.1,0.2,0.3))<<endl;
	cout<<"ele_num: "<<tetobj.eleNum()<<endl;
	tetobj.printInfo();
	cout<<"volume: "<<tetobj.eleVolume(0)<<' '<<tetobj.eleVolume(1)<<endl;
	double tetweights[4];tetobj.interpolationWeights(1,Physika::Vector<double ,3>(0.5,0.5,0.5),tetweights);
	cout<<"interplation(0.5,0.5,0.5):"<<endl;
	cout<<tetweights[0]<<' '<<tetweights[1]<<' '<<tetweights[2]<<' '<<tetweights[3]<<endl;
	cout<<endl;

	cout<<"quadMesh part: "<<endl;
	double quadvp[8]={0,0,0,1,1,0,1,1};
	unsigned int elequad[4]={0,2,3,1};
	QuadMesh<double> quadobj(4,quadvp,1,elequad);
	cout<<"contain point(0.1,0.2): "<<quadobj.containsVertex(0,Physika::Vector<double,2>(0.1,0.2))<<endl;
	cout<<"ele_num: "<<quadobj.eleNum()<<endl;
	quadobj.printInfo();
	cout<<"volume: "<<quadobj.eleVolume(0)<<endl;
	double quadweights[4];quadobj.interpolationWeights(0,Physika::Vector<double ,2>(0.5,0.3),quadweights);
	cout<<"interplation(0.5,0.3):"<<endl;
	cout<<quadweights[0]<<' '<<quadweights[1]<<' '<<quadweights[2]<<' '<<quadweights[3]<<endl;
	cout<<endl;

	cout<<"cubicMesh part:"<<endl;
	double cubicvp[24]={0,0,0 ,1,0,0, 1,1,0 ,0,1,0 ,0,0,1 ,1,0,1, 1,1,1 ,0,1,1};
	unsigned int elecubic[8]={0,1,2,3,4,5,6,7};
	CubicMesh<double> cubicobj(8,cubicvp,1,elecubic);
	cubicobj.printInfo();
	cout<<"contain point(2,2,2): "<<cubicobj.containsVertex(0,Physika::Vector<double, 3>(2,2,2))<<endl;
	cout<<"vert_num: "<<cubicobj.vertNum()<<endl;
	cout<<"volume: "<<cubicobj.eleVolume(0)<<endl;
	double cubicweights[8];
	cubicobj.interpolationWeights(0,Physika::Vector<double,3>(2,2,1),cubicweights);
	cout<<"interplation(2,2,1):"<<endl;
	for(int i=0;i<8;++i)cout<<cubicweights[i]<<' ';
	cout<<endl;
	cout<<"regionNum:"<<cubicobj.regionNum()<<endl;
	vector<unsigned int>re1;
	re1.push_back(16);re1.push_back(17);re1.push_back(5);
	cubicobj.addRegion(string("head"),re1);
	for(int i=0;i<cubicobj.regionNum();++i)
	{
		cout<<cubicobj.regionName(i)<<" ";
	}
	cout<<endl;

	getchar();

    return 0;
}
