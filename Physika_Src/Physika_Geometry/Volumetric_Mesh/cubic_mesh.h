/*
 * @file  cubic_mesh.h
 * @brief hexahedral mesh class.
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


#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_CUBIC_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_CUBIC_MESH_H_

#include "Physika_Geometry/Volumetric_Mesh/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,3>;

template <typename Scalar>
class CubicMesh: public VolumetricMesh<Scalar,3>
{
public:
    CubicMesh();
    CubicMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements);
    ~CubicMesh();
    void printInfo() const;
    int eleVolume(int ele_idx) const;
    bool containsVertex(int ele_idx, const Vector<Scalar,3> &pos) const;
    void interpolationWeights(int ele_idx, const Vector<Scalar,3> &pos, Scalar *weights) const;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_CUBIC_MESH_H_
















