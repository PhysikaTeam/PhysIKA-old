/*
 * @file obj_mesh_io.h 
 * @brief load and save mesh to a script file of mesh object for PovRay.
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_SURFACE_MESH_IO_POV_MESH_IO_H_
#define PHYSIKA_IO_SURFACE_MESH_IO_POV_MESH_IO_H_

namespace Physika{


template <typename Scalar> class SurfaceMesh;

/*
 * load/save mesh to PovRay mesh2 object script
 * Note: material information is lost in .povmesh, while texture information stays
 */

template <typename Scalar>
class PovMeshIO
{
public:
    PovMeshIO(){}
    ~PovMeshIO(){}
    //memory of mesh is preallocated by caller
    //return true if succeed, otherwise return false
    // load a mesh from a .povmesh script file.
    static bool load(const std::string &filename, SurfaceMesh<Scalar> *mesh);
    //memory of mesh is preallocated by caller
    //return true if succeed, otherwise return false
    // save a mesh to a .povmesh script file.
    static bool save(const std::string &filename, const SurfaceMesh<Scalar> *mesh);

protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_IO_SURFACE_MESH_IO_POV_MESH_IO_H_
