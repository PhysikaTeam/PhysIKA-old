/*
 * @file surface_mesh.h 
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_

#include <vector>
using std::vector;

namespace Physika{

//These forward declaration can be useful if #include is too much.
template <typename Scalar> class Triangle;
template <typename Scalar> class Vertex;
template <typename Scalar> class Edge;

template <typename Scalar>
class SurfaceMesh
{
public:
    SurfaceMesh();
    ~SurfaceMesh();
    void compute_normals();
    unsigned int get_number_of_vertex();
    unsigned int get_number_of_edge();
    unsigned int get_number_of_triangle();

    vector<Triangle<Scalar>*> triangles;
    vector<Vertex<Scalar>*> vertices;
    vector<Edge<Scalar>*> edges;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_
