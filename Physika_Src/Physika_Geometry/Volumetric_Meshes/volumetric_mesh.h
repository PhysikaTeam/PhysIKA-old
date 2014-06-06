/*
 * @file  volumetric_mesh.h
 * @brief Abstract parent class of volumetric mesh, for FEM simulation.
 *        The mesh not necessarily 3D, although with name VolumetricMesh.
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_H_

#include <cstring>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

template <typename Scalar, int Dim>
class VolumetricMesh
{
public:
    VolumetricMesh();
    VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, int vert_per_ele);
    VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele_list);//for volumetric mesh with arbitrary element type
    virtual ~VolumetricMesh();
    inline int vertNum() const{return vert_num_;}
    inline int eleNum() const{return ele_num_;}
    inline bool isUniformElementType() const{return uniform_ele_type_;}
    inline int eleVertNum(int ele_idx) const{return uniform_ele_type_?(*vert_per_ele_):(vert_per_ele_[ele_idx]);}
    Vector<Scalar,Dim> vertPos(int vert_idx) const;
    Vector<Scalar,Dim> eleVertPos(int ele_idx, int vert_idx) const;
    virtual void printInfo() const=0;
    virtual Scalar eleVolume(int ele_idx) const=0;
    virtual bool containsVertex(int ele_idx, const Vector<Scalar,Dim> &pos) const=0;
    virtual void interpolationWeights(int ele_idx, const Vector<Scalar,Dim> &pos, Scalar *weights) const=0;
protected:
    //if all elements have same number of vertices, vert_per_ele is pointer to one integer representing the vertex number per element
    //otherwise it's pointer to a list of vertex number per element
    void init(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele, bool uniform_ele_type);
protected:
    int vert_num_;
    Scalar *vertices_;
    int ele_num_;
    int *elements_;
    int *vert_per_ele_;
    bool uniform_ele_type_;        
};

}  //end of namespace Physika

//implementations
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh-inl.h"

#endif//PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_H_
