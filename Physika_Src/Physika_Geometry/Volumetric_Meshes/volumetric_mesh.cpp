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

#include <cstdlib>
#include <algorithm>
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
VolumetricMesh<Scalar,Dim>::VolumetricMesh(const VolumetricMesh<Scalar,Dim> &volumetric_mesh)
{
    this->vertices_ = volumetric_mesh.vertices_;
    this->ele_num_ = volumetric_mesh.ele_num_;
    this->elements_ = volumetric_mesh.elements_;
    this->uniform_ele_type_ = volumetric_mesh.uniform_ele_type_;
    this->vert_per_ele_ = volumetric_mesh.vert_per_ele_;
    (this->regions_).clear();
    for(unsigned int i = 0; i < volumetric_mesh.regions_.size(); ++i)
    {
        Region *src_region = volumetric_mesh.regions_[i];
        Region *region = new Region(*src_region);
        (this->regions_).push_back(region);
    }
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>::~VolumetricMesh()
{
    for(unsigned int i = 0; i < regions_.size(); ++i)
        if(regions_[i])
            delete regions_[i];
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>& VolumetricMesh<Scalar,Dim>::operator= (const VolumetricMesh<Scalar,Dim> &volumetric_mesh)
{
    this->vertices_ = volumetric_mesh.vertices_;
    this->ele_num_ = volumetric_mesh.ele_num_;
    this->elements_ = volumetric_mesh.elements_;
    this->uniform_ele_type_ = volumetric_mesh.uniform_ele_type_;
    this->vert_per_ele_ = volumetric_mesh.vert_per_ele_;
    (this->regions_).clear();
    for(unsigned int i = 0; i < volumetric_mesh.regions_.size(); ++i)
    {
        Region *src_region = volumetric_mesh.regions_[i];
        Region *region = new Region(*src_region);
        (this->regions_).push_back(region);
    }
    return *this;
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::eleVertNum(unsigned int ele_idx) const
{
    if(ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return uniform_ele_type_?vert_per_ele_[0]:vert_per_ele_[ele_idx];
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::eleVertIndex(unsigned int ele_idx, unsigned int local_vert_idx) const
{   
    if(ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int max_vert_num = uniform_ele_type_ ? vert_per_ele_[0] : vert_per_ele_[ele_idx];
    if(local_vert_idx >= max_vert_num)
    {
        std::cerr<<"vert_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int ele_idx_start = eleStartIdx(ele_idx);
    return elements_[ele_idx_start + local_vert_idx];
}

template <typename Scalar, int Dim>
int VolumetricMesh<Scalar,Dim>::eleRegionIndex(unsigned int ele_idx) const
{
    for(unsigned int i = 0; i < regions_.size(); ++i)
    {
        const vector<unsigned int> &region_elements = regions_[i]->elements();
        vector<unsigned int>::const_iterator iter = find(region_elements.begin(),region_elements.end(),ele_idx);
        if(iter != region_elements.end())
            return static_cast<int>(iter - region_elements.begin());
    }
    return -1;
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionNum() const
{
    return regions_.size();
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> VolumetricMesh<Scalar,Dim>::vertPos(unsigned int vert_idx) const
{
    if(vert_idx>=this->vertNum())
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertices_[vert_idx];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> VolumetricMesh<Scalar,Dim>::eleVertPos(unsigned int ele_idx, unsigned int local_vert_idx) const
{
    unsigned int global_vert_idx = eleVertIndex(ele_idx,local_vert_idx);
    return vertPos(global_vert_idx);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::eleVertPos(unsigned int ele_idx, std::vector<Vector<Scalar,Dim> > &positions) const
{
    if(ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    positions.clear();
    for(unsigned int i = 0; i < this->eleVertNum(ele_idx); ++i)
        positions.push_back(this->eleVertPos(ele_idx,i));
}

template <typename Scalar, int Dim>
string VolumetricMesh<Scalar,Dim>::regionName(unsigned int region_idx) const
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    return regions_[region_idx]->name();
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionEleNum(unsigned int region_idx) const
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    return regions_[region_idx]->elementNum();
}

template <typename Scalar, int Dim>
unsigned int VolumetricMesh<Scalar,Dim>::regionEleNum(const string &region_name) const
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

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::regionElements(unsigned int region_idx, vector<unsigned int> &elements) const
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    elements = regions_[region_idx]->elements();
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::regionElements(const string &region_name, vector<unsigned int> &elements) const
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

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::setVertPos(unsigned int vert_idx, const Vector<Scalar,Dim> &vert_pos)
{
    if(vert_idx>=this->vertNum())
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertices_[vert_idx] = vert_pos;
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::setEleVertPos(unsigned int ele_idx, unsigned int local_vert_idx, const Vector<Scalar,Dim> &vert_pos)
{
    unsigned int global_vert_idx = eleVertIndex(ele_idx,local_vert_idx);
    setVertPos(global_vert_idx,vert_pos);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::setEleVertPos(unsigned int ele_idx, const vector<Vector<Scalar,Dim> > &positions)
{    
    if(ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int ele_vert_num = this->eleVertNum(ele_idx); 
    if(positions.size() < ele_vert_num)
    {
        std::cerr<<"Insufficient number of vertex positions provided!\n";
        std::exit(EXIT_FAILURE);
    }
    for(unsigned int i = 0; i < ele_vert_num; ++i)
        setEleVertPos(ele_idx,i,positions[i]);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::renameRegion(unsigned int region_idx, const string &name)
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    regions_[region_idx]->setName(name);
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
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    delete regions_[region_idx];
    vector<Region*>::iterator iter = regions_.begin()+region_idx;
    regions_.erase(iter);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeRegion(const string &region_name)
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

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::addVertex(const Vector<Scalar,Dim> &vertex)
{
    vertices_.push_back(vertex);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeVertex(unsigned int vert_idx)
{
    if(vert_idx>=this->vertNum())
    {
        std::cerr<<"vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    typename vector<Vector<Scalar,Dim> >::iterator iter = vertices_.begin() + vert_idx;
    vertices_.erase(iter);
    //make the new mesh data valid: remove its reference in elements and adjust the vertex indices of other elements
    vector<unsigned int> new_elements;
    for(unsigned int i = 0; i < ele_num_; ++i)
    {
        unsigned int ele_start_idx = this->eleStartIdx(i);
        unsigned int reference_count = 0;
        for(int j = 0; j < this->eleVertNum(i); ++j)
        {
            unsigned idx = ele_start_idx + j;
            if(elements_[idx]==vert_idx)  //the element contains this vertex, reference is removed from the elements
                ++reference_count;
            else if(elements_[idx]>vert_idx) //reference to vertex with greater indices decreases by 1
                new_elements.push_back(elements_[idx]-1);
            else //reference to vertex with smaller indices, do nothing
                new_elements.push_back(elements_[idx]);
        }
        //for nonuniform element type mesh, decrease the vertex number of this element
        //do nothing if it's uniform element type, only that the mesh is not valid anymore
        if(uniform_ele_type_==false)
            vert_per_ele_[i] -= reference_count;
    }
    elements_ = new_elements; //update the elements
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
    if(ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    //first remove this element from all regions
    if(this->regionNum()>1)
    {
        for(int i = 0; i < regions_.size(); ++i)
        {
            PHYSIKA_ASSERT(regions_[i]);
            int ele_local_idx = regions_[i]->elementLocalIndex(ele_idx);
            if(ele_local_idx>=0)
                regions_[i]->removeElementAtIndex(ele_local_idx);
        }
    }
    //then remove this element
    unsigned int ele_start_idx = eleStartIdx(ele_idx);
    unsigned int ele_vert_num = eleVertNum(ele_idx);
    typename vector<unsigned int>::iterator iter_begin = elements_.begin() + ele_start_idx;
    typename vector<unsigned int>::iterator iter_end = iter_begin + ele_vert_num;
    elements_.erase(iter_begin,iter_end);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::addElementInRegion(unsigned int region_idx, unsigned int new_ele_idx)
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    if(new_ele_idx>=this->ele_num_)
    {
        std::cerr<<"element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    regions_[region_idx]->addElement(new_ele_idx);
}

template <typename Scalar, int Dim>
void VolumetricMesh<Scalar,Dim>::removeElementInRegion(unsigned int region_idx, unsigned int ele_idx_in_region)
{
    if(region_idx>=this->regionNum())
    {
        std::cerr<<"region_idx out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    if(ele_idx_in_region>=this->regionEleNum(region_idx))
    {
        std::cerr<<"ele_idx_in_region out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    PHYSIKA_ASSERT(regions_[region_idx]);
    regions_[region_idx]->removeElement(ele_idx_in_region);
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
    //create the default region
    vector<unsigned int> region_data(ele_num_);
    for(unsigned int i = 0; i < ele_num_; ++i)
        region_data[i] = i;
    Region *all_elements = new Region(string("AllElements"),region_data);
    regions_.clear();
    regions_.push_back(all_elements);
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

//explicit instantiations
template class VolumetricMesh<float,2>;
template class VolumetricMesh<float,3>;
template class VolumetricMesh<double,2>;
template class VolumetricMesh<double,3>;

}  //end of namespace Physika
