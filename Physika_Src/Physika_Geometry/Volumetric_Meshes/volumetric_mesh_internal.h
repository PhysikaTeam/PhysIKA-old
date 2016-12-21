/*
 * @file  volumetric_mesh_internal.h
 * @brief Internal types for VolumetricMesh
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_

#include <string>
#include <vector>
#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

namespace VolumetricMeshInternal{

//region class, used to represent a set of elements
class Region
{
public:
    Region();
    Region(const std::string &region_name, const std::vector<unsigned int> &elements);
    Region(const Region &region);
    ~Region();
    Region& operator= (const Region &region);
    const std::string& name() const;
    void setName(const std::string &new_name);
    unsigned int elementNum() const;
    const std::vector<unsigned int>& elements() const;
    void addElement(unsigned int new_ele_idx); //add a new element at the end of this region
    void removeElement(unsigned int ele_idx);  //remove the given element from the region, print error if element not in the region
    void removeElementAtIndex(unsigned int ele_idx_in_region); //remove the ele_idx_in_region th element from the region
    int elementLocalIndex(unsigned int ele_idx); //if contains the element, return its local index; otherwise return -1
protected:
    std::string name_;
    std::vector<unsigned int> elements_;
};

//element type of volumetric mesh
enum ElementType{
    TRI, //2D triangle
    QUAD, //2D quad
    TET, //3D tet
    CUBIC, //3D cubic
    NON_UNIFORM //non uniform 
};

//provide an operator() to compare two std::vector
//use element-wise dictionary comparison
//vectors must be equal in size
template <typename Scalar>
class CompareVector
{
public:
    //return true if v1 < v2
    bool operator()(const std::vector<Scalar> &v1, const std::vector<Scalar> &v2) const
    {
        unsigned int size1 = v1.size();
        unsigned int size2  = v2.size();
        if(size1 != size2)
            throw PhysikaException("CompareVector cannot compare vectors of different sizes!");
        for(unsigned int i = 0; i < size1; ++i)
        {
            if(v1[i] < v2[i])
                return true;
            if(v1[i] > v2[i])
                return false;
        }
        return false;
    }
};

} //end of namespace VolumetricMeshInternal

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_
