/*
 * @file obj_mesh_io.h 
 * @Basic obj_IO, load a mesh or write a mesh to a obj file. simply without texture.
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

#ifndef PHYSIKA_IO_MESH_IO_OBJ_MESH_IO_H_
#define PHYSIKA_IO_MESH_IO_OBJ_MESH_IO_H_

#include "Physika_IO/Mesh_IO/mesh_io_base.h"


namespace Physika{

template <typename Scalar>
class ObjMeshIO:public MeshIOBase<Scalar>
{
public:
    ObjMeshIO();
    ~ObjMeshIO();

    // read a mesh from a obj file.
    virtual void load(const string& filename, SurfaceMesh<Scalar> *mesh);

    // write a mesh to a obj file.
    virtual void save(const string& filename, SurfaceMesh<Scalar> *mesh);

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_MESH_IO_OBJ_IO_H_
