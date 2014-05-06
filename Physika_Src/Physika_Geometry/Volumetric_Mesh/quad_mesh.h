/*
 * @file  quad_mesh.h
 * @brief Counterpart of hexahedral mesh in 2D.
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_QUAD_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_QUAD_MESH_H_

#include "Physika_Geometry/Volumetric_Mesh/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,2>;

template <typename Scalar>
class QuadMesh: public VolumetricMesh<Scalar,2>
{
public:
    QuadMesh();
    QuadMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements);
    ~QuadMesh();
    void printInfo() const;
    int eleVolume(int ele_idx) const;
    bool containsVertex(int ele_idx, const Vector<Scalar,2> &pos) const;
    void interpolationWeights(int ele_idx, const Vector<Scalar,2> &pos, Scalar *weights) const;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_QUAD_MESH_H_
















