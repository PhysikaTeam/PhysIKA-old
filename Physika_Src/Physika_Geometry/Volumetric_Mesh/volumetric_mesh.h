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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_VOLUMETRIC_MESH_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_VOLUMETRIC_MESH_H_

#include <cstring>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/Vector_3d.h"

namespace Physika{

template <typename Scalar, int Dim>
class VolumetricMesh
{
public:
    VolumetricMesh();
    //if elements have same number of vertices (default value), vert_per_ele is pointer to one integer representing the vertex number per element
    //otherwise it's pointer to a list of vertex number per element
    VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele, bool uniform_ele_type=true);
    ~VolumetricMesh();
    inline int vertNum() const{return vert_num_;}
    inline int eleNum() const{return ele_num_;}
    inline bool isUniformElementType() const{return uniform_ele_type_;}
    inline int eleVertNum(int ele_idx) const{return uniform_ele_type_?(*vert_per_ele_):(vert_per_ele_[ele_idx]);}
protected:
    int vert_num_;
    Scalar *vertices_;
    int ele_num_;
    int *elements_;
    int *vert_per_ele_;
    bool uniform_ele_type_;
};

//implementations
template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh()
    :vert_num_(0),vertices_(NULL),ele_num_(0),elements_(NULL),vert_per_ele_(NULL),uniform_ele_type_(false)
{
}

template <typename Scalar, int Dim>
    VolumetricMesh<Scalar,Dim>::VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele, bool same_vert_num_per_ele)
    :vert_num_(vert_num),ele_num_(ele_num),uniform_ele_type_(uniform_ele_type)
{
    vertices_ = new Scalar[vert_num_*Dim];
    memcpy(vertices_,vertices,vert_num_*Dim*sizeof(Scalar));
    int elements_total_num = 0;
    if(uniform_ele_type_)
    {
	vert_per_ele_ = new int[1];
	*vert_per_ele_ = *vert_per_ele;
	elements_total_num = ele_num_*(*vert_per_ele_);
    }
    else
    {
	vert_per_ele_ = new int[ele_num_];
	memcpy(vert_per_ele_,vert_per_ele,ele_num_*sizeof(int));
	for(int i = 0; i < ele_num_; ++i)
	    elements_total_num += vert_per_ele_[i];
    }
    elements_ = new int[elements_total_num];
    memcpy(elements_,elements,elements_total_num*sizeof(int));
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::~VolumetricMesh()
{
    if(vertices_)
	delete vertices_;
    if(elements_)
	delete elements_;
    if(vert_per_ele_)
	delete vert_per_ele_;
}

}  //end of namespace Physika

#endif//PHYSIKA_GEOMETRY_VOLUMETRIC_MESH_VOLUMETRIC_MESH_H_
