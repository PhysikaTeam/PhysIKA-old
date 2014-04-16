/*
 * @file mesh_IO_base.h 
 * @Basic mesh loader, every 3d model IO of mesh will inherit from this.
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

#ifndef PHYSIKA_IO_3DMODEL_MESH_IO_BASE_H_
#define PHYSIKA_IO_3DMODEL_MESH_IO_BASE_H_

#include <string>
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

using std::string;

namespace Physika{

template <typename Scalar>
class MeshIOBase
{
public:
    MeshIOBase();
    virtual ~MeshIOBase(){};

    virtual void load(const string& filename, SurfaceMesh<Scalar> *mesh)=0;

    virtual void save(const string& filename, SurfaceMesh<Scalar> *mesh)=0;

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_3DMODEL_MESH_IO_BASE_H_
