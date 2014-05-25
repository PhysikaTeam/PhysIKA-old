/*
 * @file  tri_mesh.h
 * @brief Counterpart of Tetrahedral mesh in 2D.
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TRI_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TRI_MESH_H_

#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,2>;

template <typename Scalar>
class TriMesh: public VolumetricMesh<Scalar,2>
{
public:
    TriMesh();
    TriMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements);
    ~TriMesh();
    void printInfo() const;
    Scalar eleVolume(int ele_idx) const;
    bool containsVertex(int ele_idx, const Vector<Scalar,2> &pos) const;
    void interpolationWeights(int ele_idx, const Vector<Scalar,2> &pos, Scalar *weights) const;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TRI_MESH_H_
