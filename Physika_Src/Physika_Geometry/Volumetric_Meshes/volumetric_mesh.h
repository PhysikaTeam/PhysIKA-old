/*
 * @file  volumetric_mesh.h
 * @brief Abstract parent class of volumetric mesh, for FEM simulation.
 *        The mesh is not necessarily 3D, although with the name VolumetricMesh.
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

#include <vector>
#include <string>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

namespace VolumetricMeshInternal{

//internal class, used to represent the set of elements
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

/*
 * The elements of volumetric mesh can optionally belong to diffferent regions.
 * We assume the regions have unique names.
 */

template <typename Scalar, int Dim>
class VolumetricMesh
{
public:
    VolumetricMesh();  //construct an empty volumetric mesh
    //construct one-region mesh with given data
    VolumetricMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, unsigned int vert_per_ele);
    VolumetricMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, const unsigned int *vert_per_ele_list);//for volumetric mesh with arbitrary element type
    virtual ~VolumetricMesh();
    
    inline unsigned int vertNum() const{return vertices_.size();}
    inline unsigned int eleNum() const{return ele_num_;}
    inline bool isUniformElementType() const{return uniform_ele_type_;}
    unsigned int eleVertNum(unsigned int ele_idx) const;
    unsigned int eleVertIndex(unsigned int ele_idx, unsigned int vert_idx) const; //return the global vertex index of a specific vertex of the element
    unsigned int regionNum() const;
    const Vector<Scalar,Dim>& vertPos(unsigned int vert_idx) const;
    const Vector<Scalar,Dim>& eleVertPos(unsigned int ele_idx, unsigned int vert_idx) const;
    std::string regionName(unsigned int region_idx) const;
    void renameRegion(unsigned int region_idx, const std::string &name);
    unsigned int regionEleNum(unsigned int region_idx) const;
    unsigned int regionEleNum(const std::string &region_name) const; //print error and return 0 if no region with the given name
    //given the region index or name, return the elements of this region
    void regionElements(unsigned int region_idx, std::vector<unsigned int> &elements) const;
    void regionElements(const std::string &region_name, std::vector<unsigned int> &elements) const; //print error and return empty elements if no region with the given name
    void addRegion(const std::string &name, const std::vector<unsigned int> &elements);
    void removeRegion(unsigned int region_idx);
    void removeRegion(const std::string &region_name);  //print error if no region with the given name

    //virtual methods
    virtual void printInfo() const=0;
    virtual VolumetricMeshInternal::ElementType elementType() const=0;
    virtual int eleVertNum() const=0; //only valid when uniform_ele_type_ is true, return the number of vertices per element, otherwise return -1
    virtual Scalar eleVolume(unsigned int ele_idx) const=0;
    virtual bool containsVertex(unsigned int ele_idx, const Vector<Scalar,Dim> &pos) const=0;
    virtual void interpolationWeights(unsigned int ele_idx, const Vector<Scalar,Dim> &pos, Scalar *weights) const=0; 
protected:
    //if all elements have same number of vertices, vert_per_ele is pointer to one integer representing the vertex number per element
    //otherwise it's pointer to a list of vertex number per element
    void init(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements, const unsigned int *vert_per_ele, bool uniform_ele_type);
protected:
    std::vector<Vector<Scalar,Dim> > vertices_;
    unsigned int ele_num_;
    std::vector<unsigned int> elements_; //vertex index of each element in order (0-index)
    bool uniform_ele_type_; //whether Elements are of uniform type (same number of vertices)
    //if uniform_ele_type_ = true, vert_per_ele_ contains only 1 integer, which is the number of vertices per element
    //if uniform_ele_type_ = false, vert_per_ele_ is a list of integers, corresponding to each element
    std::vector<unsigned int> vert_per_ele_;
    //regions_ is empty if all elements belong to one region
    //otherwise regions_ contains list of regions  
    std::vector<VolumetricMeshInternal::Region*> regions_;
};

}  //end of namespace Physika

#endif//PHYSIKA_GEOMETRY_VOLUMETRIC_MESHES_VOLUMETRIC_MESH_H_
