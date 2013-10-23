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

class Triangle;
class Vertex;
class Edge;

class SurfaceMesh
{
public:
    SurfaceMesh();
    ~SurfaceMesh();


    vector<Triangle*> triangles;
    vector<Vertex*> vertices;
    vector<Edge*> edges;

    void compute_normals();

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_
