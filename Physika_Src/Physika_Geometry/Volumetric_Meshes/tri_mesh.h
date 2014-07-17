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

#include <vector>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,2>;

template <typename Scalar>
class TriMesh: public VolumetricMesh<Scalar,2>
{
public:
    TriMesh(); //construct an empty TriMesh
    TriMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements); //construct TriMesh with given data
    TriMesh(const TriMesh<Scalar> &tri_mesh);
    ~TriMesh();
    TriMesh<Scalar>& operator=(const TriMesh<Scalar> &tri_mesh);
    void printInfo() const;
    VolumetricMeshInternal::ElementType elementType() const;
    unsigned int eleVertNum() const;
    Scalar eleVolume(unsigned int ele_idx) const;
    bool containsVertex(unsigned int ele_idx, const Vector<Scalar,2> &pos) const;
    void interpolationWeights(unsigned int ele_idx, const Vector<Scalar,2> &pos, std::vector<Scalar> &weights) const;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TRI_MESH_H_
