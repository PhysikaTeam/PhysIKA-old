/*
 * @file surface_mesh.cpp 
 * @Basic surface mesh class.
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

#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Geometry/Surface_Mesh/triangle.h"
#include "Physika_Geometry/Surface_Mesh/edge.h"
#include "Physika_Geometry/Surface_Mesh/vertex.h"

namespace Physika{
	
SurfaceMesh::SurfaceMesh()
{
	//int a;
}

SurfaceMesh::~SurfaceMesh()
{

}

void SurfaceMesh::compute_normals()
{
	const int nbtriangles = triangles.size();
	const int nbvertices = vertices.size();
	const int nbedges = edges.size();

	//for face normals;
	for(int i = 0; i < nbtriangles; ++i)
	{
		Triangle *t = triangles[i];
		t->compute_normal();
		t->normal.normalize();
	}

	//for edges: angle weighted nomrmal;
	for (int i = 0; i < nbedges; ++i)
	{
		Edge *e = edges[i];
		assert(e != NULL);
		e->normal = 0.0f;
		assert(e->triangles[0] != NULL && e->triangles[1] != NULL);
		e->normal += e->triangles[0]->normal + e->triangles[1]->normal;
		float length = e->normal.norm();
	}

	//for vertices;
	for(int i = 0; i < nbtriangles; i++)
	{
		Triangle *t = triangles[i];
		assert(t != NULL);
		for (int j=0; j<3; j++) {
			Vector3f e1 = *t->vertices[(j+1)%3] - *t->vertices[j];
			Vector3f e2 = *t->vertices[(j+2)%3] - *t->vertices[j];
			e1.normalize();
			e2.normalize();
			float weight = acos(e1.dot(e2));
			t->vertices[j]->normal += weight * t->normal;
		}
	}
	for (int i = 0; i < nbvertices; ++i)
	{
		Vertex *v = vertices[i];
		assert(v != NULL);
		v->normal.normalize();
	}


}

} //end of namespace Physika
