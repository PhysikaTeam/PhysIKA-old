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
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

using namespace std;
using Physika::CollidableObject;
using Physika::MeshBasedCollidableObject;
using Physika::SurfaceMesh;
using Physika::Vector;


int main()
{
	cout<<"123"<<endl;
	MeshBasedCollidableObject<double, 3>* pObject;
	pObject = new MeshBasedCollidableObject<double, 3>();
//	if(pObject->getObjectType()== CollidableObject<double, 3>::ObjectType::MESH_BASED)
//		cout<<pObject->getObjectType()<<endl;

	
	//pObject->getMesh() = new SurfaceMesh<double>();

	system("pause");

    return 0;
}
