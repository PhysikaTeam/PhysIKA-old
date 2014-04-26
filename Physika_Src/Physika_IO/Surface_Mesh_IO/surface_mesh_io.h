/*
 * @file surface_mesh_IO.h 
 * @brief surface mesh loader/saver, load/save surface mesh from/to file.
 *        dynamically choose different loader/saver with respect to file suffix.
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

#ifndef PHYSIKA_IO_SURFACE_MESH_IO_SURFACE_MESH_IO_H_
#define PHYSIKA_IO_SURFACE_MESH_IO_SURFACE_MESH_IO_H_

#include <string>
using std::string;

namespace Physika{

template <typename Scalar> class SurfaceMesh;

template <typename Scalar>
class SurfaceMeshIO
{
public:
    SurfaceMeshIO(){}
    ~SurfaceMeshIO(){}
    //memory of mesh is preallocated by caller
    static void load(const string& filename, SurfaceMesh<Scalar> *mesh);
    static void save(const string& filename, SurfaceMesh<Scalar> *mesh);

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_SURFACE_MESH_IO_SURFACE_MESH_IO_H_



