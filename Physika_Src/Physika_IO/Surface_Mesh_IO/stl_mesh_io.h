/*
 * @file stl_mesh_io.h 
 * @brief load and save mesh to a stl file.
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

#ifndef PHYSIKA_IO_SURFACE_MESH_IO_STL_MESH_IO_H_
#define PHYSIKA_IO_SURFACE_MESH_IO_STL_MESH_IO_H_

#include <string>

namespace Physika{

template <typename Scalar> class SurfaceMesh;

template <typename Scalar>
class StlMeshIO
{
public:
    StlMeshIO(){}
    ~StlMeshIO(){}

    // load a mesh from a stl file.
    static bool load(const std::string& filename, SurfaceMesh<Scalar> *mesh);
    // save a mesh to a stl file.
    static bool save(const std::string& filename, SurfaceMesh<Scalar> *mesh);

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_SURFACE_MESH_IO_STL_MESH_IO_H_






