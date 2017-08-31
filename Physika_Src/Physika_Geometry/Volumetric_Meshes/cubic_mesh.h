/*
 * @file  cubic_mesh.h
 * @brief hexahedral mesh class.
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


#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_CUBIC_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_CUBIC_MESH_H_

#include <vector>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

/*
 * the order of the vertices matters:
 *
 *                     7--------6
 *                    / |       / |
 *                   4-|------5 |
 *                   |  3-----|- 2
 *                   |/        | /
 *                  0---------1
 */

template <typename Scalar> class Vector<Scalar,3>;

template <typename Scalar>
class CubicMesh: public VolumetricMesh<Scalar,3>
{
public:
    CubicMesh(); //construct an empty CubicMesh
    CubicMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements); //construct CubicMesh with given data
    CubicMesh(const CubicMesh<Scalar> &cubic_mesh);
    ~CubicMesh();
    CubicMesh<Scalar>& operator= (const CubicMesh<Scalar> &cubic_mesh);
    CubicMesh<Scalar>* clone() const;
    void printInfo() const;
    VolumetricMeshInternal::ElementType elementType() const;
    unsigned int eleVertNum() const;
    Scalar eleVolume(unsigned int ele_idx) const;
    bool containPoint(unsigned int ele_idx, const Vector<Scalar,3> &pos) const;
    void interpolationWeights(unsigned int ele_idx, const Vector<Scalar,3> &pos, std::vector<Scalar> &weights) const;
protected:
    virtual void generateBoundaryInformation();
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_CUBIC_MESH_H_
