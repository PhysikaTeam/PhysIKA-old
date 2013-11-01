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
    /* Constructions */
    SurfaceMesh();
    ~SurfaceMesh();
    
    /* Get and Set */
    inline vector<Edge<Scalar>*>& edges() { return edges_;}
    inline vector<Vertex<Scalar>*> & vertices() { return vertices_;}
    inline vector<Triangle<Scalar>*> & triangles() { return triangles_;}
    inline unsigned int getNumVertex() const { return vertices_.size(); }
    inline unsigned int getNumEdge() const { return edges_.size(); }
    inline unsigned int getNumTriangle() const {return triangles_.size(); }

    /* Some functions */
    void computeNormals();


protected:
    vector<Triangle<Scalar>*> triangles_;
    vector<Vertex<Scalar>*> vertices_;
    vector<Edge<Scalar>*> edges_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_
