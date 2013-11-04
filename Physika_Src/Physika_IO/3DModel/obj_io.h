/*
 * @file obj_IO.h 
 * @Basic obj_IO, load a mesh or write a mesh to a obj file. simply without texture.
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

#ifndef PHYSIKA_IO_3DMODEL_OBJ_IO_H_
#define PHYSIKA_IO_3DMODEL_OBJ_IO_H_

#include "Physika_IO/3DModel/mesh_io_base.h"


namespace Physika{

template <typename Scalar>
class ObjIO:public MeshIOBase<Scalar>
{
public:
    ObjIO();
    ~ObjIO();

    // read a mesh from a obj file.
    virtual void read(const string& filename, SurfaceMesh<Scalar> *mesh);

    // write a mesh to a obj file.
    virtual void write(const string& filename, SurfaceMesh<Scalar> *mesh);

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_3DMODEL_OBJ_IO_H_
