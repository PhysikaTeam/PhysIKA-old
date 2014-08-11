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

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_GROUP_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_GROUP_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Boundary_Meshes/material.h"
#include "Physika_Geometry/Boundary_Meshes/face.h"

namespace Physika{

namespace SurfaceMeshInternal{

/*
 * we assume groups can be identified by their names
 */

template <typename Scalar>
class FaceGroup
{
public:
    FaceGroup();
    ~FaceGroup();
    explicit FaceGroup(const std::string &name);
    FaceGroup(const std::string &name, const std::vector<Face<Scalar> > &faces);
    FaceGroup(const std::string &name, unsigned int material_index);
    FaceGroup(const std::string &name, unsigned int material_index, const std::vector<Face<Scalar> > &faces);
    unsigned int numFaces() const;
    const std::string& name() const;
    void setName(const std::string &name);
    const Face<Scalar>& face(unsigned int face_idx) const; 
    Face<Scalar>& face(unsigned int face_idx);
    const Face<Scalar>* facePtr(unsigned int face_idx) const;
    Face<Scalar>* facePtr(unsigned int face_idx);
    unsigned int materialIndex() const;
    void setMaterialIndex(unsigned int material_index);
    void addFace(const Face<Scalar> &face);
    void removeFace(unsigned int face_idx);
protected:
    std::string name_;
    unsigned int material_index_; //materil index in the mesh
    std::vector<Face<Scalar> > faces_;
};

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_GROUP_H_
