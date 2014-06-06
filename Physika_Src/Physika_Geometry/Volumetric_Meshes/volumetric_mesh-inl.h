/*
 * @file  volumetric_mesh-inl.h
 * @brief implementation of methods defined in volumetric_mesh.h
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INL_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INL_H_

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh()
    :vert_num_(0),vertices_(NULL),ele_num_(0),elements_(NULL),vert_per_ele_(NULL),uniform_ele_type_(false)
{
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, int vert_per_ele)
{
    init(vert_num,vertices,ele_num,elements,&vert_per_ele,true);
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele_list)
{
    init(vert_num,vertices,ele_num,elements,vert_per_ele_list,false);
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::~VolumetricMesh()
{
    if(vertices_)
	    delete[] vertices_;
    if(elements_)
        delete[] elements_;
    if(vert_per_ele_)
	    delete[] vert_per_ele_;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> VolumetricMesh<Scalar,Dim>::vertPos(int vert_idx) const
{
    if((vert_idx<0) || (vert_idx>=this->vert_num_))
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Vector<Scalar,Dim> pos;
    for(int i = 0; i < Dim; ++i)
        pos[i] = vertices_[Dim*vert_idx+i];
    return pos;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> VolumetricMesh<Scalar,Dim>::eleVertPos(int ele_idx, int vert_idx) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    int ele_idx_start = 0;
    if(uniform_ele_type_)
    {
		if((vert_idx<0) || (vert_idx >= (*vert_per_ele_) ))
		{
			std::cerr<<"vert_idx out of range\n";
			std::exit(EXIT_FAILURE);
		}
        ele_idx_start = ele_idx*(*vert_per_ele_);
    }
    else
    {
		if((vert_idx<0) || (vert_idx >= vert_per_ele_[ele_idx]))
		{
			std::cerr<<"vert_idx out of range\n";
			std::exit(EXIT_FAILURE);
		}
        for(int i = 0; i < ele_idx; ++i)
            ele_idx_start += vert_per_ele_[i];
    }
    int global_vert_idx = elements_[ ele_idx_start +vert_idx];
    return vertPos(global_vert_idx);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::init(int vert_num, const Scalar *vertices, int ele_num, const int *elements, const int *vert_per_ele, bool uniform_ele_type)
{
    vert_num_ = vert_num;
    ele_num_ = ele_num;
    uniform_ele_type_ = uniform_ele_type;
    vertices_ = new Scalar[vert_num_*Dim];
    PHYSIKA_ASSERT(vertices_);
    memcpy(vertices_,vertices,vert_num_*Dim*sizeof(Scalar));
    int elements_total_num = 0;
    if(uniform_ele_type_)
    {
        vert_per_ele_ = new int[1];
        PHYSIKA_ASSERT(vert_per_ele_);
        *vert_per_ele_ = *vert_per_ele;
        elements_total_num = ele_num_*(*vert_per_ele_);
    }
    else
    {
        vert_per_ele_ = new int[ele_num_];
        PHYSIKA_ASSERT(vert_per_ele_);
        memcpy(vert_per_ele_,vert_per_ele,ele_num_*sizeof(int));
        for(int i = 0; i < ele_num_; ++i)
            elements_total_num += vert_per_ele_[i];
    }
    elements_ = new int[elements_total_num];
    PHYSIKA_ASSERT(elements_);
    memcpy(elements_,elements,elements_total_num*sizeof(int));
}

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INL_H_
