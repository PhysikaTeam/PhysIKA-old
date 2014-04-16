/*
 * @file stl_mesh_io.h 
 * @brief load and save mesh to a stl file.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_MESH_IO_STL_MESH_IO_H_
#define PHYSIKA_IO_MESH_IO_STL_MESH_IO_H_

#include "Physika_IO/Mesh_IO/mesh_io_base.h"


namespace Physika{

template <typename Scalar>
class StlMeshIO:public MeshIOBase<Scalar>
{
public:
    StlMeshIO();
    ~StlMeshIO();

    // load a mesh from a stl file.
    static void load(const string& filename, SurfaceMesh<Scalar> *mesh);

    // save a mesh to a stl file.
    static void save(const string& filename, SurfaceMesh<Scalar> *mesh);

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_MESH_IO_STL_MESH_IO_H_






