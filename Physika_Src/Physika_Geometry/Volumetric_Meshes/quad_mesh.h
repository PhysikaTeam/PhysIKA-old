/*
 * @file  quad_mesh.h
 * @brief Counterpart of hexahedral mesh in 2D.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_QUAD_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_QUAD_MESH_H_

#include <vector>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

template <typename Scalar> class Vector<Scalar,2>;

template <typename Scalar>
class QuadMesh: public VolumetricMesh<Scalar,2>
{
public:
    QuadMesh();  //construct an empty QuadMesh
    QuadMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements);  //construct QuadMesh with given data
    QuadMesh(const QuadMesh<Scalar> &quad_mesh);
    ~QuadMesh();
    QuadMesh<Scalar>& operator= (const QuadMesh<Scalar> &quad_mesh);
    QuadMesh<Scalar>* clone() const;
    void printInfo() const;
    VolumetricMeshInternal::ElementType elementType() const;
    unsigned int eleVertNum() const;
    Scalar eleVolume(unsigned int ele_idx) const;
    bool containPoint(unsigned int ele_idx, const Vector<Scalar,2> &pos) const;
    void interpolationWeights(unsigned int ele_idx, const Vector<Scalar,2> &pos, std::vector<Scalar> &weights) const;
protected:
    virtual void generateBoundaryInformation();
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_QUAD_MESH_H_
