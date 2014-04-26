/*
 * @file group.h 
 * @brief face group of 3d surface mesh
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_GROUP_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_GROUP_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Surface_Mesh/material.h"
#include "Physika_Geometry/Surface_Mesh/face.h"
using std::string;
using std::vector;

namespace Physika{

namespace SurfaceMeshInternal{

template <typename Scalar>
class Group
{
public:
    Group();
    ~Group();
    explicit Group(const string &name);
    Group(const string &name, const vector<Face<Scalar> > &faces);
    Group(const string &name, unsigned int material_index);
    Group(const string &name, unsigned int material_index, const vector<Face<Scalar> > &faces);
    unsigned int numFaces() const;
    const string& name() const;
    void setName(const string &name);
    const Face<Scalar>& face(unsigned int face_idx) const; 
    Face<Scalar>& face(unsigned int face_idx);
    const Face<Scalar>* facePtr(unsigned int face_idx) const;
    Face<Scalar>* facePtr(unsigned int face_idx);
    unsigned int materialIndex() const;
    void setMaterialIndex(unsigned int material_index);
    void addFace(const Face<Scalar> &face);
    void removeFace(unsigned int face_idx);
protected:
    string name_;
    unsigned int material_index_; //materil index in the mesh
    vector<Face<Scalar> > faces_;
};

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_GROUP_H_











