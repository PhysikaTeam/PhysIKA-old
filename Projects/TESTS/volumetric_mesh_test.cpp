/*
 * @file volumetric_mesh_test.cpp 
 * @brief Test the various types of volumetric meshes.
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
#include "Physika_Geometry/Volumetric_Mesh/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Mesh/tet_mesh.h"
using namespace std;
using Physika::VolumetricMesh;
using Physika::TetMesh;

int main()
{
    VolumetricMesh<float,3> *vol_mesh_ptr;
    TetMesh<float> tet_mesh;
    vol_mesh_ptr = &tet_mesh;
    vol_mesh_ptr->info();
    return 0;
}
