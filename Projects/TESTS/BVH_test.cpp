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
 * contact me at mikepkucs@gmail.com if any question
 */

#include <iostream>
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

using namespace std;
using Physika::CollidableObject;
using Physika::MeshBasedCollidableObject;


int main()
{
	cout<<"123"<<endl;
	CollidableObject<double, 3>* pObject;
	pObject = new MeshBasedCollidableObject<double, 3>();

	system("pause");

    return 0;
}
