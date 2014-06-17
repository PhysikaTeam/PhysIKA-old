/*
 * @file  volumetric_mesh_internal.h
 * @brief Internal types for VolumetricMesh
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

#ifndef PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_
#define PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_

#include <string>
#include <vector>

namespace Physika{

namespace VolumetricMeshInternal{

//region class, used to represent a set of elements
class Region
{
public:
    Region();
    Region(const std::string &region_name, const std::vector<unsigned int> &elements);
    ~Region();
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

} //end of namespace VolumetricMeshInternal

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_INTERNAL_H_

