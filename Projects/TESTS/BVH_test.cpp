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

#include <iostream>
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Geometry\Bounding_Volume\bvh_base.h"
#include "Physika_Geometry\Bounding_Volume\bvh_node_base.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry\Bounding_Volume\bounding_volume_kdop18.h"
#include "Physika_Geometry\Bounding_Volume\bounding_volume.h"

using namespace std;
using Physika::CollidableObject;
using Physika::MeshBasedCollidableObject;
using Physika::SurfaceMesh;
using Physika::Vector;
using Physika::BVHBase;
using Physika::BVHNodeBase;
using Physika::BoundingVolumeKDOP18;
using Physika::BoundingVolume;


int main()
{
	//MeshBasedCollidableObject<double, 3>* pObject;
	//pObject = new MeshBasedCollidableObject<double, 3>();
	//if(pObject->getObjectType()== CollidableObject<double, 3>::ObjectType::MESH_BASED)
	//	cout<<pObject->getObjectType()<<endl;

	
	//pObject->getMesh() = new SurfaceMesh<double>();

	//BVHBase<double, 3> * pBVH1 = new BVHBase<double, 3>();
	//BVHBase<double, 3> * pBVH2 = new BVHBase<double, 3>();
	//pBVH1->setRootNode(new BVHNodeBase<double, 3>());
	//pBVH2->setRootNode(new BVHNodeBase<double, 3>());
	//pBVH1->collide(pBVH2);

	BoundingVolumeKDOP18<double, 3> KDOP;
	Vector<double, 3> point1(1, 1, 1);
	Vector<double, 3> point2(1, 1, -1);
	Vector<double, 3> point3(1, -1, 1);
	Vector<double, 3> point4(1, -1, -1);
	Vector<double, 3> point5(-1, 1, 1);
	Vector<double, 3> point6(-1, 1, -1);
	Vector<double, 3> point7(-1, -1, 1);
	Vector<double, 3> point8(-1, -1, -1);
	KDOP.setBoundingVolume(point1);
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



	system("pause");

    return 0;
}
