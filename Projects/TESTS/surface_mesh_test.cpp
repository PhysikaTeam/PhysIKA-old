/*
 * @file surface_mesh_test.cpp 
 * @brief Test the surface_mesh/vertex/edge/triangle class.
 * @author Sheng Yang
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
#include "Physika_Geometry/Surface_Mesh/vertex.h"
#include "Physika_Geometry/Surface_Mesh/triangle.h"
#include "Physika_Geometry/Surface_Mesh/edge.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

using namespace std;
using Physika::Vertex;
using Physika::Edge;
using Physika::Triangle;
using Physika::SurfaceMesh;

int main()
{
	Vertex<float> vertex(1,2,3);
	Vertex<float> vertex1(2,3,4);
	//cout<<vertex<<endl;
	//cout<<vertex1<<endl;
	//cout<<vertex+vertex1<<endl;
	//cout<<vertex-vertex1<<endl;

	Edge<float> edge;
	Triangle<double> trianges;
	SurfaceMesh<float> mesh;
	mesh.computeNormals();
	int a;
	cin>>a;
}