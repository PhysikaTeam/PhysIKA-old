/*
 * @file  volumetric_mesh.cpp
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
using std::string;
using std::vector;

namespace Physika{

using VolumetricMeshInternal::Region;
using VolumetricMeshInternal::ElementType;
using VolumetricMeshInternal::TRI;
using VolumetricMeshInternal::QUAD;
using VolumetricMeshInternal::TET;
using VolumetricMeshInternal::CUBIC;
using VolumetricMeshInternal::NON_UNIFORM;

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh()
    :ele_num_(0),uniform_ele_type_(true)
{
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, unsigned int vert_per_ele)
{
    init(vert_num,vertices,ele_num,elements,&vert_per_ele,true);
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::VolumetricMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, const unsigned int *vert_per_ele_list)
{
    init(vert_num,vertices,ele_num,elements,vert_per_ele_list,false);
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::~VolumetricMesh()
{
    for(unsigned int i = 0; i < regions_.size(); ++i)
        if(regions_[i])
            delete regions_[i];
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::eleVertNum(unsigned int ele_idx) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return uniform_ele_type_?vert_per_ele_[0]:vert_per_ele_[ele_idx];
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::eleVertIndex(unsigned int ele_idx, unsigned int vert_idx) const
{   
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int max_vert_num = uniform_ele_type_ ? vert_per_ele_[0] : vert_per_ele_[ele_idx];
    if((vert_idx<0) || (vert_idx >= max_vert_num))
    {
        std::cerr<<"vert_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int ele_idx_start = eleStartIdx(ele_idx);
    return elements_[ele_idx_start +vert_idx];
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionNum() const
{
    return regions_.empty()?1:regions_.size();
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim>& VolumetricMesh<Scalar,Dim>::vertPos(unsigned int vert_idx) const
{
    if((vert_idx<0) || (vert_idx>=this->vertNum()))
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertices_[vert_idx];
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim>& VolumetricMesh<Scalar,Dim>::eleVertPos(unsigned int ele_idx, unsigned int vert_idx) const
{
    int global_vert_idx = eleVertIndex(ele_idx,vert_idx);
    return vertPos(global_vert_idx);
}

template <typename Scalar, int Dim>
string VolumetricMesh<Scalar,Dim>::regionName(unsigned int region_idx) const
{
    if(region_idx<0||region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    if(this->regionNum()==1) //only one region
        return string("AllElements");
    else  //multiple regions
    {
        PHYSIKA_ASSERT(regions_[region_idx]);
        return regions_[region_idx]->name();
    }
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionEleNum(unsigned int region_idx) const
{
    if(region_idx<0||region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    if(this->regionNum()==1) //only one region
        return this->eleNum();
    else
    {
        PHYSIKA_ASSERT(regions_[region_idx]);
        return regions_[region_idx]->elementNum();
    }
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionEleNum(const string &region_name) const
{
    if(this->regionNum()==1&&region_name==string("AllElements"))
        return this->eleNum();
    else
    {
        for(unsigned int i = 0; i < regions_.size(); ++i)
        {
            PHYSIKA_ASSERT(regions_[i]);
            if(regions_[i]->name()==region_name)
                return regionEleNum(i);
        }
        std::cerr<<"There's no region with the name: "<<region_name<<".\n";
        return 0;
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::regionElements(unsigned int region_idx, vector<unsigned int> &elements) const
{
    if(region_idx<0||region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    if(this->regionNum()==1)  //only one region
    {
        elements.resize(this->eleNum());
        for(int i = 0; i < elements.size(); ++i)
            elements[i] = i;
    }
    else
    {
        PHYSIKA_ASSERT(regions_[region_idx]);
        elements = regions_[region_idx]->elements();
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::regionElements(const string &region_name, vector<unsigned int> &elements) const
{
    if(this->regionNum()==1 && region_name==string("AllElements"))
        regionElements(0,elements);
    else
    {
        for(unsigned int i = 0; i < regions_.size(); ++i)
        {
            PHYSIKA_ASSERT(regions_[i]);
            if(regions_[i]->name()==region_name)
            {
                regionElements(i,elements);
                return;
            }
        }
        std::cerr<<"There's no region with the name: "<<region_name<<".\n";
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::renameRegion(unsigned int region_idx, const string &name)
{
    if(region_idx<0||region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    if(this->regionNum()==1) //only one region
        std::cout<<"Cannot rename the defualt AllElements regions.\n";
    else
    {
        PHYSIKA_ASSERT(regions_[region_idx]);
        regions_[region_idx]->setName(name);
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::addRegion(const string &name, const vector<unsigned int> &elements)
{
    Region *new_region = new Region(name,elements);
    PHYSIKA_ASSERT(new_region);
    regions_.push_back(new_region);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeRegion(unsigned int region_idx)
{
    if(region_idx<0||region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range\n";
        std::exit(EXIT_FAILURE);
    }
    if(this->regionNum()==1) //only one region
    {
        std::cerr<<"Cannot remove the default AllElements region.\n";
    }
    else
    {
        PHYSIKA_ASSERT(regions_[region_idx]);
        delete regions_[region_idx];
        vector<Region*>::iterator iter = regions_.begin()+region_idx;
        regions_.erase(iter);
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeRegion(const string &region_name)
{
    if(this->regionNum()==1 && region_name==string("AllElements")) //only one region
        std::cerr<<"Cannot remove the default AllElements region.\n";
    else
    {
        for(unsigned int i = 0; i < regions_.size(); ++i)
        {
            PHYSIKA_ASSERT(regions_[i]);
            if(regions_[i]->name()==region_name)
            {
                removeRegion(i);
                return;
            }
        }
        std::cerr<<"There's no region with the name: "<<region_name<<".\n";
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::addVertex(const Vector<Scalar,Dim> &vertex)
{
    vertices_.push_back(vertex);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeVertex(unsigned int vert_idx)
{
    if((vert_idx<0) || (vert_idx>=this->vertNum()))
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    typename vector<Vector<Scalar,Dim> >::iterator iter = vertices_.begin() + vert_idx;
    vertices_.erase(iter);
    //TO DO: make the new mesh data valid
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::addElement(const vector<unsigned int> &element)
{
    bool valid_ele_vert_num = true;
    unsigned int new_ele_vert_num = element.size();
    ElementType element_type = this->elementType();
    //vertex number of the new element must be valid
    //i.e. , cannot change uniform element type to non-uniform element type
    switch(element_type)
    {
    case TRI:
        valid_ele_vert_num = (new_ele_vert_num == 3);
        break;
    case QUAD:
        valid_ele_vert_num = (new_ele_vert_num == 4);
        break;
    case TET:
        valid_ele_vert_num = (new_ele_vert_num == 4);
        break;
    case CUBIC:
        valid_ele_vert_num = (new_ele_vert_num == 8);
        break;
    case NON_UNIFORM:
        valid_ele_vert_num = true;
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
    }
    if(valid_ele_vert_num)
    {
        //TO DO: we could check whether the vertex indices of the new element is valid
        elements_.insert(elements_.end(),element.begin(),element.end());
        ++ele_num_;
    }
    else
    {
        std::cerr<<"Invalid vertex number for the new element!\n";
        std::exit(EXIT_FAILURE);
    }
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeElement(unsigned int ele_idx)
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    //first remove this element from all regions
    if(this->regionNum()>1)
    {
        //TO DO
    }
    //then remove this element
    unsigned int ele_start_idx = eleStartIdx(ele_idx);
    unsigned int ele_vert_num = eleVertNum(ele_idx);
    typename vector<unsigned int>::iterator iter_begin = elements_.begin() + ele_start_idx;
    typename vector<unsigned int>::iterator iter_end = iter_begin + ele_vert_num;
    elements_.erase(iter_begin,iter_end);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::init(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, const unsigned int *vert_per_ele, bool uniform_ele_type)
{
    vertices_.resize(vert_num);
    for(unsigned int i = 0; i < vertices_.size(); ++i)
    {
        vertices_[i] = Vector<Scalar,Dim>();
        for(int j = 0; j < Dim; ++j)
            vertices_[i][j] = vertices[Dim*i+j];
    }
    uniform_ele_type_ = uniform_ele_type;
    ele_num_ = ele_num;
    PHYSIKA_ASSERT(vert_per_ele);
    if(uniform_ele_type_)
    {
        vert_per_ele_.resize(1);
        vert_per_ele_[0] = vert_per_ele[0];
        elements_.resize(ele_num_*vert_per_ele_[0]);
        for(unsigned int i = 0; i < elements_.size(); ++i)
            elements_[i] = elements[i];
    }
    else
    {
        vert_per_ele_.resize(ele_num_);
        unsigned int total_ele_vert_num = 0;
        for(unsigned int i = 0; i < ele_num_; ++i)
        {
            vert_per_ele_[i] = vert_per_ele[i];
            total_ele_vert_num += vert_per_ele_[i];
        }
        elements_.clear();
        for(unsigned int i = 0; i < total_ele_vert_num; ++i)
            elements_.push_back(elements[i]);
    }
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::eleStartIdx(unsigned int ele_idx) const
{
    PHYSIKA_ASSERT(ele_idx>=0);
    PHYSIKA_ASSERT(ele_idx<ele_num_);
    unsigned int ele_idx_start = 0;
    if(uniform_ele_type_)
        ele_idx_start = ele_idx*vert_per_ele_[0];
    else
    {
        for(int i = 0; i < ele_idx; ++i)
            ele_idx_start += vert_per_ele_[i];
    }
    return ele_idx_start;
}
////////////////////////////////////////////////Implementation of Region/////////////////////////////////////////////////////
namespace VolumetricMeshInternal{
Region::Region()
{
}

Region::Region(const string &region_name, const vector<unsigned int> &elements)
{
    name_ = region_name;
    elements_ =elements;
}

Region::~Region()
{
}

const string& Region::name() const
{
    return name_;
}

void Region::setName(const string &new_name)
{
    name_ = new_name;
}

unsigned int Region::elementNum() const
{
    return elements_.size();
}

const vector<unsigned int>& Region::elements() const
{
    return elements_;
}

}  //end of namespace VolumetricMeshInternal

//explicit instantiations
template class VolumetricMesh<float,2>;
template class VolumetricMesh<float,3>;
template class VolumetricMesh<double,2>;
template class VolumetricMesh<double,3>;

}  //end of namespace Physika
