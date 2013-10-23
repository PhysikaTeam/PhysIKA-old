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

template <typename Scalar>
SurfaceMesh<Scalar>::SurfaceMesh()
{
	//int a;
}

template <typename Scalar>
SurfaceMesh<Scalar>::~SurfaceMesh()
{

}

template <typename Scalar>
void SurfaceMesh<Scalar>::compute_normals()
{
    const int nbtriangles = triangles.size();
    const int nbvertices = vertices.size();
    const int nbedges = edges.size();

    //for face normals;
    for(int i = 0; i < nbtriangles; ++i)
    {
        Triangle<Scalar> *t = triangles[i];
       	t->compute_normal();
       	t->normal.normalize();
    }

    //for edges: angle weighted nomrmal;
    for (int i = 0; i < nbedges; ++i)
    {
	Edge<Scalar> *e = edges[i];
	assert(e != NULL);
       	e->normal = 0.0f;
       	assert(e->triangles[0] != NULL && e->triangles[1] != NULL);
       	e->normal += e->triangles[0]->normal + e->triangles[1]->normal;
       	Scalar length = e->normal.norm();
    }

    //for vertices;
    for(int i = 0; i < nbtriangles; i++)
    {
       	Triangle<Scalar> *t = triangles[i];
       	assert(t != NULL);
       	for (int j=0; j<3; j++)
        {
       	    Vector3D<Scalar> e1 = t->vertices[(j+1)%3]->position - t->vertices[j]->position;
	        Vector3D<Scalar> e2 = t->vertices[(j+2)%3]->position - t->vertices[j]->position;
	        e1.normalize();
	        e2.normalize();
	        Scalar weight = acos(e1.dot(e2));
            t->vertices[j]->normal += weight * t->normal;
	 }
    }
    for (int i = 0; i < nbvertices; ++i)
    {
       	Vertex<Scalar> *v = vertices[i];
       	assert(v != NULL);
       	v->normal.normalize();
    }


}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::get_number_of_vertex()
{
	return this->vertices.size()	;
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::get_number_of_edge()
{
	return this->edges.size()	;
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::get_number_of_triangle()
{
	return this->triangles.size()	;
}


template class SurfaceMesh<float>;
template class SurfaceMesh<double>;

} //end of namespace Physika
