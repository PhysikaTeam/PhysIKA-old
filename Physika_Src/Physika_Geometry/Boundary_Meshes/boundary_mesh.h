/*
 * @file boundary_mesh.h 
 * @brief virtual base class of SurfaceMesh and Polygon
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

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_BOUNDARY_MESH_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_BOUNDARY_MESH_H_

namespace Physika{

class BoundaryMesh
{
public:
    BoundaryMesh(){}
    virtual ~BoundaryMesh(){}
    virtual unsigned int dims() const = 0;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_BOUNDARY_MESH_H_
