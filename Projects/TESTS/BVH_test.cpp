/*
 * @file volumetric_mesh_test.cpp 
 * @brief Test the various types of volumetric meshes.
 * @author Mike Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include<iostream>
#include<string>
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/Timer/timer.h"

using namespace std;
using namespace Physika;


int main()
{

	BoundingVolumeKDOP18<double, 3> KDOP;
	Vector<double, 3> point1(1, 1, 1);
	Vector<double, 3> point2(1, 1, -1);
	Vector<double, 3> point3(1, -1, 1);
	Vector<double, 3> point4(1, -1, -1);
	Vector<double, 3> point5(-1, 1, 1);
	Vector<double, 3> point6(-1, 1, -1);
	Vector<double, 3> point7(-1, -1, 1);
	Vector<double, 3> point8(-1, -1, -1);
	KDOP.unionWith(point1);
	KDOP.unionWith(point2);
	KDOP.unionWith(point3);
	KDOP.unionWith(point4);
	KDOP.unionWith(point5);
	KDOP.unionWith(point6);
	KDOP.unionWith(point7);
	KDOP.unionWith(point8);

	Vector<double, 3> point(-1, 1, 0.999);
	if(KDOP.isInside(point))
		cout<<"in"<<endl;

	Timer timer;
	
    SurfaceMesh<double> mesh_ball;
    if(!ObjMeshIO<double>::load(string("E:/Physika/ball_high.obj"), &mesh_ball))
		exit(1);
	
	MeshBasedCollidableObject<double, 3>* pObject1 = new MeshBasedCollidableObject<double, 3>();
	pObject1->setMesh(&mesh_ball);
	pObject1->transform().setPosition(Vector<double, 3>(10.15, 10.15, 10.15));
	//pObject1->transform().setPosition(Vector<double, 3>(1.05, 1.05, 1.05));
	//pObject1->transform().setPosition(Vector<double, 3>(0, 0.5, 0));
	ObjectBVH<double, 3> * pBVH1 = new ObjectBVH<double, 3>();
	pBVH1->setCollidableObject(pObject1);
	
	MeshBasedCollidableObject<double, 3>* pObject2 = new MeshBasedCollidableObject<double, 3>();
	pObject2->setMesh(&mesh_ball);
	ObjectBVH<double, 3> * pBVH2 = new ObjectBVH<double, 3>();
	pBVH2->setCollidableObject(pObject2);

	CollisionDetectionResult<double, 3> result;
	result.resetCollisionResults();

	timer.startTimer();
	if(pBVH1->collide(pBVH2, result))
		cout<<"collide"<<endl;
	timer.stopTimer();
	cout<<timer.getElapsedTime()<<endl;
	cout<<result.numberPCS()<<endl;
	cout<<result.numberCollision()<<endl;

	system("pause");

    return 0;
}
