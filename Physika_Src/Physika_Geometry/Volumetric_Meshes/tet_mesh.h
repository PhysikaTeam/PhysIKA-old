/*
 * @file  tet_mesh.h
 * @brief Tetrahedral mesh class.
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TET_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TET_MESH_H_

#include <vector>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,3>;

template <typename Scalar>
class TetMesh: public VolumetricMesh<Scalar,3>
{
public:
    TetMesh();  //construct an empty TetMesh
    TetMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements); //construct TetMesh with given data
    TetMesh(const TetMesh<Scalar> &tet_mesh);
    ~TetMesh();
    TetMesh<Scalar>& operator= (const TetMesh<Scalar> &tet_mesh);
    TetMesh<Scalar>* clone() const;
    void printInfo() const;
    VolumetricMeshInternal::ElementType elementType() const;
    unsigned int eleVertNum() const;
    Scalar eleVolume(unsigned int ele_idx) const;
    bool containPoint(unsigned int ele_idx, const Vector<Scalar,3> &pos) const;
    void interpolationWeights(unsigned int ele_idx, const Vector<Scalar,3> &pos, std::vector<Scalar> &weights) const;
protected:
    //helper method for interpolationWeights()
    Scalar getTetDeterminant(const Vector<Scalar,3> &a, const Vector<Scalar,3> &b, const Vector<Scalar,3> &c, const Vector<Scalar,3> &d) const;
};

}//end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_TET_MESH_H_
